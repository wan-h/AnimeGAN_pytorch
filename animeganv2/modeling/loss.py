# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from animeganv2.configs import cfg
from torch.nn import functional as F
from .utils import gram, rgb2yuv, prepare_feature_extract, color_2_gray

def init_loss(model_backbone, real_images_color, generated):
    fake = generated
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))
    fake_feature_map = model_backbone(prepare_feature_extract(fake))
    loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')
    return loss * cfg.MODEL.COMMON.WEIGHT_G_CON

def g_loss(model_backbone, real_images_color, style_images_gray, generated, generated_logit):
    fake = generated
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))
    fake_feature_map = model_backbone(prepare_feature_extract(fake))
    fake_feature_map_gray = model_backbone(prepare_feature_extract(color_2_gray(fake)))
    anime_feature_map = model_backbone(prepare_feature_extract(style_images_gray))

    c_loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')

    if cfg.MODEL.LOSS.S_LOSS_COLOR2GRAY:
        s_loss = F.l1_loss(gram(anime_feature_map), gram(fake_feature_map_gray), reduction='mean')
    else:
        s_loss = F.l1_loss(gram(anime_feature_map), gram(fake_feature_map), reduction='mean')

    real_images_color_yuv = rgb2yuv(real_images_color)
    fake_yuv = rgb2yuv(fake)
    color_loss = F.l1_loss(real_images_color_yuv[..., 0], fake_yuv[..., 0], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[..., 1], fake_yuv[..., 1], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[..., 2], fake_yuv[..., 2], reduction='mean')


    dh_input, dh_target = fake[:, :, :-1, :], fake[:, :, 1:, :]
    dw_input, dw_target = fake[:, :, :, :-1], fake[:, :, :, 1:]
    tv_loss = F.mse_loss(dh_input, dh_target, reduction='mean') + \
              F.mse_loss(dw_input, dw_target, reduction='mean')

    loss_func = cfg.MODEL.COMMON.GAN_TYPE
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = - torch.mean(generated_logit)
    elif loss_func == 'lsgan':
        fake_loss = torch.mean(torch.square(generated_logit - 1.0))
    elif loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = F.binary_cross_entropy_with_logits(generated_logit, torch.ones_like(generated_logit), reduction='mean')
    elif loss_func == 'hinge':
        fake_loss = - torch.mean(generated_logit)
    else:
        raise NotImplementedError
    return cfg.MODEL.COMMON.WEIGHT_G_CON * c_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_STYLE * s_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_COLOR * color_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_TV * tv_loss + \
           cfg.MODEL.COMMON.WEIGHT_ADV_G * fake_loss

from  torch import autograd
from torch.autograd import Variable
def gradient_panalty(model_discriminator, style_images_color, generted):
    loss_func = cfg.MODEL.COMMON.GAN_TYPE
    if loss_func not in ['dragan', 'wgan-gp', 'wgan-lp']:
        return 0
    if loss_func == 'dragan':
        eps = torch.empty_like(style_images_color).uniform_(0, 1)
        x_var = style_images_color.var()
        x_std = torch.sqrt(x_var)
        generted = style_images_color + 0.5 * x_std * eps
    b, c, h, w = style_images_color.shape
    device = style_images_color.device
    alpha = torch.Tensor(b, 1, 1, 1).uniform_(0, 1)
    alpha = alpha.expand(b, c, h, w).to(device)
    interpolated = style_images_color + alpha * (generted - style_images_color)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)
    # calculate probability of interpolated examples
    prob_interpolated = model_discriminator(interpolated)
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True
    )[0]
    GP = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * cfg.MODEL.COMMON.LD
    return GP

def d_loss(generated_logit, anime_logit, anime_gray_logit, smooth_logit):
    loss_func = cfg.MODEL.COMMON.GAN_TYPE
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = - torch.mean(anime_logit)
        gray_loss = torch.mean(anime_gray_logit)
        fake_loss = torch.mean(generated_logit)
        real_blur_loss = torch.mean(smooth_logit)
    elif loss_func == 'lsgan':
        real_loss = torch.mean(torch.square(anime_logit - 1.0))
        gray_loss = torch.mean(torch.square(anime_gray_logit))
        fake_loss = torch.mean(torch.square(generated_logit))
        real_blur_loss = torch.mean(torch.square(smooth_logit))
    elif loss_func == 'gan' or loss_func == 'dragan':
        real_loss = F.binary_cross_entropy_with_logits(anime_logit, torch.ones_like(anime_logit), reduction='mean')
        gray_loss = F.binary_cross_entropy_with_logits(anime_gray_logit, torch.zeros_like(anime_gray_logit), reduction='mean')
        fake_loss = F.binary_cross_entropy_with_logits(generated_logit, torch.zeros_like(generated_logit), reduction='mean')
        real_blur_loss = F.binary_cross_entropy_with_logits(smooth_logit, torch.zeros_like(smooth_logit), reduction='mean')
    elif loss_func == 'hinge':
        real_loss = torch.mean(torch.relu(1.0 - anime_logit))
        gray_loss = torch.mean(torch.relu(1.0 + anime_gray_logit))
        fake_loss = torch.mean(torch.relu(1.0 + generated_logit))
        real_blur_loss = torch.mean(torch.relu(1.0 + smooth_logit))
    else:
        raise NotImplementedError
    return cfg.MODEL.COMMON.WEIGHT_ADV_D * (
            cfg.MODEL.COMMON.WEIGHT_D_LOSS_REAL * real_loss +
            cfg.MODEL.COMMON.WEIGHT_D_LOSS_FAKE * fake_loss +
            cfg.MODEL.COMMON.WEIGHT_D_LOSS_GRAY * gray_loss +
            cfg.MODEL.COMMON.WEIGHT_D_LOSS_BLUR * real_blur_loss
    )
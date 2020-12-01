# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from animeganv2.configs import cfg
from torch.nn import functional as F
from .utils import gram, rgb2yuv, prepare_feature_extract

def init_loss(model_backbone, model_generator, real_images_color):
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))
    fake = model_generator(real_images_color)
    fake_feature_map = model_backbone(prepare_feature_extract(fake))
    loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')
    return loss * cfg.MODEL.COMMON.WEIGHT_G_CON

def g_loss(model_backbone, model_generator, model_discriminator, real_images_color, style_images_gray):
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))
    fake = model_generator(real_images_color)
    fake_feature_map = model_backbone(prepare_feature_extract(fake))
    anime_feature_map = model_backbone(prepare_feature_extract(style_images_gray))

    c_loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')

    s_loss = F.l1_loss(gram(anime_feature_map), gram(fake_feature_map), reduction='mean')

    real_images_color_yuv = rgb2yuv(real_images_color)
    fake_yuv = rgb2yuv(fake)
    color_loss = F.l1_loss(real_images_color_yuv[:, 0, :, :], fake_yuv[:, 0, :, :], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[:, 1, :, :], fake_yuv[:, 1, :, :], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[:, 2, :, :], fake_yuv[:, 2, :, :], reduction='mean')


    dh_input, dh_target = fake[:, :, :-1, :], fake[:, :, 1:, :]
    dw_input, dw_target = fake[:, :, :, :-1], fake[:, :, :, 1:]
    tv_loss = F.mse_loss(dh_input, dh_target, reduction='mean') / dh_input.numel() + \
              F.mse_loss(dw_input, dw_target, reduction='mean') / dw_input.numel()

    loss_func = cfg.MODEL.COMMON.GAN_TYPE
    generated_logit = model_discriminator(fake)
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = - torch.mean(generated_logit)
    elif loss_func == 'lsgan':
        fake_loss = torch.mean(torch.square(generated_logit - 1.0))
    elif loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = F.cross_entropy(F.sigmoid(generated_logit), torch.ones_like(generated_logit), reduction='mean')
    elif loss_func == 'hinge':
        fake_loss = - torch.mean(generated_logit)
    else:
        raise NotImplementedError
    return cfg.MODEL.COMMON.WEIGHT_G_CON * c_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_STYLE * s_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_STYLE * color_loss + \
           cfg.MODEL.COMMON.WEIGHT_G_TV * tv_loss + \
           cfg.MODEL.COMMON.WEIGHT_ADV_G * fake_loss


def d_loss(model_generator, model_discriminator, real_images_color, style_images_color, style_images_gray, smooth_images_gray):
    anime_logit = model_discriminator(style_images_color)
    anime_gray_logit = model_discriminator(style_images_gray)
    generated = model_generator(real_images_color)
    generated_logit = model_discriminator(generated)
    smooth_logit = model_discriminator(smooth_images_gray)
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
        real_loss = F.cross_entropy(F.sigmoid(anime_logit), torch.ones_like(anime_logit), reduction='mean')
        gray_loss = F.cross_entropy(F.sigmoid(anime_gray_logit), torch.zeros_like(anime_gray_logit), reduction='mean')
        fake_loss = F.cross_entropy(F.sigmoid(generated_logit), torch.zeros_like(generated_logit), reduction='mean')
        real_blur_loss = F.cross_entropy(F.sigmoid(smooth_logit), torch.zeros_like(smooth_logit), reduction='mean')
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
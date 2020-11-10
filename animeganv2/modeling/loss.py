# coding: utf-8
# Author: wanhui0729@gmail.com

from animeganv2.configs import cfg
from torch.nn import functional as F

def init_loss(model_backbone, model_generator, real_images):
    real_feature_map = model_backbone(real_images)
    fake = model_generator(real_images)
    fake_feature_map = model_backbone(fake)
    loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')
    return loss * cfg.MODEL.COMMON.WEIGHT_CON
# coding: utf-8
# Author: wanhui0729@gmail.com

from .backbone import build_backbone
from .generator import build_generator
from .discriminator import build_discriminator


def build_model(cfg):
    backbone = build_backbone(cfg)
    generator = build_generator(cfg)
    discriminator = build_discriminator(cfg)
    return backbone, generator, discriminator
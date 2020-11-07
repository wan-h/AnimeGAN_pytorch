# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn
from animeganv2.modeling import registry
from animeganv2.modeling.utils import Conv2DNormLReLU, InvertedRes_Block

class G_Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.A = nn.Sequential(
            Conv2DNormLReLU(in_channels, 32, kernel_size=7, padding=3),
            Conv2DNormLReLU(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2DNormLReLU(64, 64, kernel_size=3, padding=1)
        )
        self.B = nn.Sequential(
            Conv2DNormLReLU(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
        )
        self.C = nn.Sequential(
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
            InvertedRes_Block(128, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            Conv2DNormLReLU(256, 128, kernel_size=3, padding=1)
        )
        self.D = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
            Conv2DNormLReLU(128, 128, kernel_size=1, padding=1)
        )
        self.E = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv2DNormLReLU(128, 64, kernel_size=3, padding=1),
            Conv2DNormLReLU(64, 64, kernel_size=1, padding=1),
            Conv2DNormLReLU(64, 32, kernel_size=7, padding=3)
        )
        self.F = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1)
        )
        self.fake = nn.Tanh()

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        x = self.C(x)
        x = self.D(x)
        x = self.E(x)
        x = self.F(x)
        x = self.fake(x)
        return x

@registry.GENERATOR.register("Base-256")
def build_base_generator(cfg):
    in_channels = cfg.MODEL.GENERATOR.IN_CHANNELS
    return G_Net(in_channels)

def build_generator(cfg):
    assert cfg.MODEL.GENERATOR.BODY in registry.GENERATOR, \
        f"cfg.MODEL.GENERATOR.BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    return registry.GENERATOR[cfg.MODEL.GENERATOR.BODY](cfg)
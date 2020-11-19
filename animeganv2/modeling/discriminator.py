# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn
from animeganv2.modeling import registry

class D_Net(nn.Module):
    def __init__(self, in_channels, channels, n_dis):
        super().__init__()
        channels = channels // 2
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        second_list = []
        channels_in = channels
        for _ in range(n_dis):
            second_list += [
                nn.Conv2d(channels_in, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2)
            ]
            second_list += [
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(channels * 4, affine=True),
                nn.LeakyReLU(0.2)
            ]
            channels_in = channels * 4
            channels *= 2
        self.second = nn.Sequential(*second_list)

        self.third = nn.Sequential(
            nn.Conv2d(channels_in, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        return x

@registry.DISCRIMINATOR.register("Base-256")
def build_base_discriminator(cfg):
    in_channels = cfg.MODEL.DISCRIMINATOR.IN_CHANNELS
    channels = cfg.MODEL.DISCRIMINATOR.CHANNELS
    n_dis = cfg.MODEL.DISCRIMINATOR.N_DIS
    return D_Net(in_channels, channels, n_dis)

def build_discriminator(cfg):
    assert cfg.MODEL.DISCRIMINATOR.BODY in registry.DISCRIMINATOR, \
        f"cfg.MODEL.DISCRIMINATOR.BODY: {cfg.MODEL.DISCRIMINATOR.CONV_BODY} are not registered in registry"
    return registry.DISCRIMINATOR[cfg.MODEL.DISCRIMINATOR.BODY](cfg)
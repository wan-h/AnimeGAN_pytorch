# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from torch import nn as nn
from torch.nn.utils import spectral_norm
from animeganv2.modeling import registry
from animeganv2.modeling.layers import Layer_Norm

def conv_sn(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    return spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    )


class D_Net(nn.Module):
    def __init__(self, in_channels, channels, n_dis):
        super().__init__()
        channels = channels // 2
        self.first = nn.Sequential(
            conv_sn(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        second_list = []
        channels_in = channels
        for _ in range(n_dis):
            second_list += [
                conv_sn(channels_in, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2)
            ]
            second_list += [
                conv_sn(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                Layer_Norm(),
                nn.LeakyReLU(0.2)
            ]
            channels_in = channels * 4
            channels *= 2
        self.second = nn.Sequential(*second_list)

        self.third = nn.Sequential(
            conv_sn(channels_in, channels * 2, kernel_size=3, stride=1, padding=1),
            Layer_Norm(),
            nn.LeakyReLU(0.2),
            conv_sn(channels * 2, 1, kernel_size=3, stride=1, padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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
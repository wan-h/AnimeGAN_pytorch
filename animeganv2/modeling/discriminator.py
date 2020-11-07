# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn

class D_Net(nn.Module):
    def __init__(self, in_channels, n_dis):
        super().__init__()
        out_channels = in_channels // 2
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        second_list = []
        for _ in range(n_dis):
            in_channels, out_channels = out_channels, out_channels * 2
            second_list += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            ]
            in_channels, out_channels = out_channels, out_channels * 2
            second_list += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ]
        self.second = nn.Sequential(*second_list)

        in_channels, out_channels = out_channels, out_channels * 2
        self.third = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        return x

def build_discriminator(cfg):
    in_channels = cfg.MODEL.DISCRIMINATOR.IN_CHANNELS
    n_dis = cfg.MODEL.DISCRIMINATOR.N_DIS
    return D_Net(in_channels, n_dis)
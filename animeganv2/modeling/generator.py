# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn
from animeganv2.modeling import registry
from animeganv2.modeling.layers import Layer_Norm

class Conv2DNormLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode='reflect',
                              bias=bias)
        self.LayerNorm = Layer_Norm()
        self.LRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.Conv(x)
        x = self.LayerNorm(x)
        x = self.LRelu(x)
        return x

class InvertedRes_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride):
        super().__init__()
        self.add_op = (in_channels == out_channels and stride == 1)
        bottleneck_dim = round(expansion_ratio * in_channels)
        # pw
        self.pw = Conv2DNormLReLU(in_channels, bottleneck_dim, kernel_size=1)
        # dw
        self.dw = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=stride, padding=1, groups=bottleneck_dim),
            Layer_Norm(),
            nn.LeakyReLU(0.2)
        )
        # pw & linear
        self.pw_linear = nn.Sequential(
            nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1),
            Layer_Norm()
        )

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        out = self.pw_linear(out)
        if self.add_op:
            out += x
        return out

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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1)
        )
        self.E = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2DNormLReLU(128, 64, kernel_size=3, padding=1),
            Conv2DNormLReLU(64, 64, kernel_size=3, padding=1),
            Conv2DNormLReLU(64, 32, kernel_size=7, padding=3)
        )
        self.F = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        x = self.C(x)
        x = self.D(x)
        x = self.E(x)
        x = self.F(x)
        return x

@registry.GENERATOR.register("Base-256")
def build_base_generator(cfg):
    in_channels = cfg.MODEL.GENERATOR.IN_CHANNELS
    return G_Net(in_channels)

def build_generator(cfg):
    assert cfg.MODEL.GENERATOR.BODY in registry.GENERATOR, \
        f"cfg.MODEL.GENERATOR.BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    return registry.GENERATOR[cfg.MODEL.GENERATOR.BODY](cfg)
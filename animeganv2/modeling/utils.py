# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn

class Conv2DNormLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode='reflect',
                              bias=bias)
        self.Inst_Norm = nn.InstanceNorm2d(out_channels)
        self.LRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.Conv(x)
        x = self.Inst_Norm(x)
        x = self.LRelu(x)
        return x

class Layer_Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m = nn.LayerNorm(x.size()[1:])
        return m(x)

class InvertedRes_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride):
        super().__init__()
        bottleneck_dim = round(expansion_ratio * in_channels)
        # pw
        self.pw = Conv2DNormLReLU(in_channels, bottleneck_dim, kernel_size=1)
        # dw
        self.dw = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=stride, padding=1, groups=bottleneck_dim),
            Layer_Norm(),
            nn.LeakyReLU()
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
        return x + out


class DSConv(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class IRB(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class Down_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class Up_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
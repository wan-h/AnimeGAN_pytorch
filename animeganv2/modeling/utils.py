# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]

def rgb2yuv(x):
    # TODO: 这行代码的作用?
    x = (x + 1.0) / 2.0
    b, c, h, w = x.shape
    im_flat = x.view(b, c, h * w).float()
    mat = torch.stack([torch.Tensor(_rgb_to_yuv_kernel) for _ in range(b)]).to(x.device)
    temp = torch.bmm(mat, im_flat)
    out = temp.view(x.shape)
    return out

def gram(x):
    shape = x.shape
    b = shape[0]
    c = shape[1]
    x = torch.reshape(x, [b, -1, c])
    return torch.bmm(x.permute(0, 2, 1), x) / (x.numel() // b)

# Calculates the average brightness in the specified irregular image
def calculate_average_brightness(img):
    # Average value of three color channels
    R = img[..., 0].mean()
    G = img[..., 1].mean()
    B = img[..., 2].mean()

    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R

# Adjusting the average brightness of the target image to the average brightness of the source image
def adjust_brightness_from_src_to_dst(dst, src):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2

    # According to the average display brightness
    dstf = dst * brightness_difference

    # According to the average value of the three-color channel
    # dstf = dst.copy().astype(np.float32)
    # dstf[..., 0] = dst[..., 0] * (B1 / B2)
    # dstf[..., 1] = dst[..., 1] * (G1 / G2)
    # dstf[..., 2] = dst[..., 2] * (R1 / R2)

    # To limit the results and prevent crossing the border,
    # it must be converted to uint8, otherwise the default result is float32, and errors will occur.
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    return dstf

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
        return F.layer_norm(x, x.size()[1:])

class InvertedRes_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride):
        super().__init__()
        self.add_op = in_channels == out_channels
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
        if self.add_op:
            out += x
        return out
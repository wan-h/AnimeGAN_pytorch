# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
import numpy as np

yuv_from_rgb = np.array([[0.299,       0.587,       0.114],
                         [-0.14714119, -0.28886916, 0.43601035],
                         [0.61497538,  -0.51496512, -0.10001026]])

# Pretrained
# rgb
# feature_extract_mean = [0.485, 0.456, 0.406]
# feature_extract_std = [0.229, 0.224, 0.225]
feature_extract_mean = [123.68, 116.779, 103.939]

def rgbScaled(x):
    # [-1, 1] ~ [0, 1]
    return (x + 1.0) / 2.0

def rgb2yuv(x):
    x = rgbScaled(x)
    x = x.permute([0, 2, 3, 1])
    k_yuv_from_rgb = torch.from_numpy(yuv_from_rgb.T).to(x.dtype).to(x.device)
    yuv = torch.matmul(x, k_yuv_from_rgb)
    # yuv = yuv.permute([0, 3, 1, 2])
    return yuv

def gram(x):
    # [b, c, h, w] -> [b, h, w, c]
    x = x.permute([0, 2, 3, 1])
    shape = x.shape
    b = shape[0]
    c = shape[3]
    x = torch.reshape(x, [b, -1, c])
    return torch.matmul(x.permute(0, 2, 1), x) / (x.numel() // b)

def prepare_feature_extract(rgb):
    # [-1, 1] ~ [0, 255]
    rgb_scaled = rgbScaled(rgb) * 255.0
    R, G, B = torch.chunk(rgb_scaled, 3, 1)
    feature_extract_input = torch.cat(
        [
            (B - feature_extract_mean[2]),
            (G - feature_extract_mean[1]),
            (R - feature_extract_mean[0]),
        ],
        dim=1
    )
    return feature_extract_input

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
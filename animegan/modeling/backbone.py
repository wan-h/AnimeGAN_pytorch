# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from animegan.modeling import registry

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    '''
    只保留features相关层
    '''
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x

# 定制化，输出conv4_4_no_activation
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    conv_stage = 1
    inner_stage = 0
    for v in cfg:
        if v == 'M':
            conv_stage += 1
            inner_stage = 0
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            inner_stage += 1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if conv_stage == 4 and inner_stage == 4:
                layers += [conv2d]
                # 特征返回点
                break
            else:
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

@registry.BACKBONES.register("VGG19")
def build_vgg_backbones(cfg):
    backbone_weiht = cfg.MODEL.BACKBONE.WEIGHT
    model = VGG(make_layers(cfgs['E']))
    if not backbone_weiht:
        state_dict = load_state_dict_from_url(model_urls['vgg19'], progress=True)
    else:
        state_dict = torch.load(backbone_weiht)
    model.load_state_dict(state_dict, strict=False)
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.BODY in registry.BACKBONES, \
        f"cfg.MODEL.BACKBONES.BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    return registry.BACKBONES[cfg.MODEL.BACKBONE.BODY](cfg)
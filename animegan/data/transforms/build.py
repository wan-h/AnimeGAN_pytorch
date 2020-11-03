# coding: utf-8
# Author: wanhui0729@gmail.com

from . import transforms as T

def build_transforms(cfg, is_train=True):
    tansform = T.Compose(cfg, is_train)
    return tansform
# coding: utf-8
# Author: wanhui0729@gmail.com

from yacs.config import CfgNode as CN
_C = CN()

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# 训练设备类型
_C.MODEL.DEVICE = "cuda"
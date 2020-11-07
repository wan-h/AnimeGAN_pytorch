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

# -----------------------------------------------------------------------------
# BACKBONE
# -----------------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()

# -----------------------------------------------------------------------------
# DISCRIMINATOR
# -----------------------------------------------------------------------------
_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.IN_CHANNELS = 64
_C.MODEL.DISCRIMINATOR.N_DIS = 2

# -----------------------------------------------------------------------------
# GENERATOR
# -----------------------------------------------------------------------------
_C.MODEL.GENERATOR = CN()
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
_C.MODEL.BACKBONE.BODY = "VGG19"

# -----------------------------------------------------------------------------
# DISCRIMINATOR
# -----------------------------------------------------------------------------
_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.BODY = "Base-256"
_C.MODEL.DISCRIMINATOR.IN_CHANNELS = 64
_C.MODEL.DISCRIMINATOR.N_DIS = 2

# -----------------------------------------------------------------------------
# GENERATOR
# -----------------------------------------------------------------------------
_C.MODEL.GENERATOR = CN()
_C.MODEL.GENERATOR.BODY = "Base-256"
_C.MODEL.GENERATOR.IN_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.GENERATOR = CN()
_C.SOLVER.GENERATOR.BASE_LR = 0.00002
_C.SOLVER.GENERATOR.GENERATOR.STEPS = (150, 200, 250)
_C.SOLVER.GENERATOR.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.GENERATOR.WARMUP_ITERS = 500
_C.SOLVER.GENERATOR.WARMUP_METHOD = "linear"

_C.SOLVER.DISCRIMINATOR = CN()
_C.SOLVER.DISCRIMINATOR.BASE_LR = 0.00004
_C.SOLVER.DISCRIMINATOR.GENERATOR.STEPS = (150, 200, 250)
_C.SOLVER.DISCRIMINATOR.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.DISCRIMINATOR.WARMUP_ITERS = 500
_C.SOLVER.DISCRIMINATOR.WARMUP_METHOD = "linear"

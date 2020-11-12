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
# Common options
# ---------------------------------------------------------------------------- #
_C.MODEL.COMMON = CN()
_C.MODEL.COMMON.GAN_TYPE = 'lsgan'
_C.MODEL.COMMON.TRAINING_RATE = 1

_C.MODEL.COMMON.WEIGHT_ADV_G = 300.0
_C.MODEL.COMMON.WEIGHT_ADV_D = 300.0
_C.MODEL.COMMON.WEIGHT_G_CON = 1.5
_C.MODEL.COMMON.WEIGHT_G_STYLE = 2.5
_C.MODEL.COMMON.WEIGHT_G_COLOR = 10.0
_C.MODEL.COMMON.WEIGHT_G_TV = 1.0
_C.MODEL.COMMON.WEIGHT_D_LOSS_REAL = 1.7
_C.MODEL.COMMON.WEIGHT_D_LOSS_FAKE = 1.7
_C.MODEL.COMMON.WEIGHT_D_LOSS_GRAY = 1.7
_C.MODEL.COMMON.WEIGHT_D_LOSS_BLUR = 1.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset Info for training
_C.DATASETS.TRAIN = [
    {
        'factory': 'PascalVOCDataset',
        'dataDir': '/data/datasets/voc/VOC2007',
        'split': 'train'
    }
]
# List of the dataset Info for testing
_C.DATASETS.TEST = [
    {
        'factory': 'PascalVOCDataset',
        'dataDir': '/data/datasets/voc/VOC2007',
        'split': 'test'
    }
]

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCH = 100
# 单位epoch
_C.SOLVER.CHECKPOINT_PERIOD = 20
# 单位iteration
_C.SOLVER.PRINT_PERIOD = 20
# 单位epoch
_C.SOLVER.TEST_PERIOD = 10
# 单位epoch
_C.SOLVER.PRUNE_PERIOD = 1

_C.SOLVER.GENERATOR = CN()
_C.SOLVER.GENERATOR.INIT_EPOCH = 10
_C.SOLVER.GENERATOR.BASE_LR = 0.0002
_C.SOLVER.GENERATOR.GENERATOR.STEPS = (_C.SOLVER.GENERATOR.INIT_EPOCH,)
_C.SOLVER.GENERATOR.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.GENERATOR.WARMUP_ITERS = 0
_C.SOLVER.GENERATOR.WARMUP_METHOD = "constant"

_C.SOLVER.DISCRIMINATOR = CN()
_C.SOLVER.DISCRIMINATOR.BASE_LR = 0.00004
_C.SOLVER.DISCRIMINATOR.STEPS = (_C.SOLVER.MAX_EPOCH,)
_C.SOLVER.DISCRIMINATOR.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.DISCRIMINATOR.WARMUP_ITERS = 0
_C.SOLVER.DISCRIMINATOR.WARMUP_METHOD = "linear"

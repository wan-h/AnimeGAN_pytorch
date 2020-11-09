# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from .lr_scheduler import WarmupMultiStepLR

def make_optimizer(cfg, model):
    lr = cfg.SOLVER.BASE_LR
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, 0.999))
    return optimizer


def make_lr_scheduler(cfg, optimizer, epoch_size):
    return WarmupMultiStepLR(
        optimizer,
        [step * epoch_size for step in cfg.SOLVER.STEPS],
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
from .lr_scheduler import WarmupMultiStepLR

def make_optimizer_generator(cfg, model):
    lr = cfg.SOLVER.GENERATOR.BASE_LR
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, 0.999))
    return optimizer

def make_optimizer_discriminator(cfg, model):
    lr = cfg.SOLVER.DISCRIMINATOR.BASE_LR
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, 0.999))
    return optimizer


def make_lr_scheduler_generator(cfg, optimizer, epoch_size):
    return WarmupMultiStepLR(
        optimizer,
        [step * epoch_size for step in cfg.SOLVER.GENERATOR.STEPS],
        cfg.SOLVER.GENERATOR.GAMMA,
        warmup_factor=cfg.SOLVER.GENERATOR.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.GENERATOR.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.GENERATOR.WARMUP_METHOD,
    )

def make_lr_scheduler_discriminator(cfg, optimizer, epoch_size):
    return WarmupMultiStepLR(
        optimizer,
        [step * epoch_size for step in cfg.SOLVER.DISCRIMINATOR.STEPS],
        cfg.SOLVER.DISCRIMINATOR.GAMMA,
        warmup_factor=cfg.SOLVER.DISCRIMINATOR.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.DISCRIMINATOR.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.DISCRIMINATOR.WARMUP_METHOD,
    )
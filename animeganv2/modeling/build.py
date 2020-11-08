# coding: utf-8
# Author: wanhui0729@gmail.com

from .generalized import GeneralizedAnimeGan

_META_ARCHITECTURES = {
    "GeneralizedAnimeGan": GeneralizedAnimeGan,
}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
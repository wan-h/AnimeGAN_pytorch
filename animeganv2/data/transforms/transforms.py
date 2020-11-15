# coding: utf-8
# Author: wanhui0729@gmail.com

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.cfg = cfg

    def __call__(self, images):
        if self.is_train:
            assert len(images) == 3
            outputs = []
            for image in images:
                size = self.cfg.INPUT.IMG_SIZE
                image = cv2.resize(image, size)
                image_color = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                image_color[:, :, 0] += self.cfg.INPUT.PIXEL_MEAN[0]
                image_color[:, :, 1] += self.cfg.INPUT.PIXEL_MEAN[1]
                image_color[:, :, 2] += self.cfg.INPUT.PIXEL_MEAN[2]
                image_color = torch.from_numpy((image_color / 127.5 - 1.0).transpose((2, 0, 1)))
                image_gray = torch.from_numpy((image_gray / 127.5 - 1.0))
                image_gray = torch.stack([image_gray, image_gray, image_gray])
                outputs.append([image_color, image_gray])
        else:
            assert len(images) == 1
            img = images[0]
            size = self.cfg.INPUT.IMG_SIZE
            h, w = img.shape[:2]
            if h <= size[1]:
                h = size[1]
            else:
                x = h % 32
                h = h - x
            if w < size[0]:
                w = size[0]
            else:
                y = w % 32
                w = w - y
            # the cv2 resize func : dsize format is (W ,H)
            image = cv2.resize(img, (w, h))
            image_color = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_color = torch.from_numpy((image_color / 127.5 - 1.0).transpose((2, 0, 1)))
            image_gray = torch.from_numpy((image_gray / 127.5 - 1.0))
            image_gray = torch.stack([image_gray, image_gray, image_gray])
            outputs = [[image_color, image_gray]]
        return outputs
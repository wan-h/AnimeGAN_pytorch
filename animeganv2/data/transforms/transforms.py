# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
import random
import numpy as np

class Compose():
    def __init__(self, cfg, is_train):
        self.cfg = cfg
        self.is_train = is_train

    def __call__(self, images):
        outputs = []
        if self.is_train:
            assert len(images) == 3
            # real
            grayAndNormlize_real = GrayAndNormlize(mean=self.cfg.INPUT.PIXEL_MEAN)
            randomCrop_style = RandomCrop(self.cfg.INPUT.IMG_SIZE[0], self.cfg.INPUT.IMG_SIZE[1])
            crop_images = randomCrop_style(images[0])
            outputs += grayAndNormlize_real(crop_images, use_norm=False)
            # style and smooth
            randomCrop_style = RandomCrop(self.cfg.INPUT.IMG_SIZE[0], self.cfg.INPUT.IMG_SIZE[1])
            grayAndNormlize_style = GrayAndNormlize(mean=self.cfg.INPUT.PIXEL_MEAN)
            crop_images = randomCrop_style(images[1:])
            outputs += grayAndNormlize_style(crop_images)
        else:
            assert len(images) == 1
            resize_test = Resize(self.cfg.INPUT.IMG_SIZE)
            grayAndNormlize_test = GrayAndNormlize(mean=self.cfg.INPUT.PIXEL_MEAN)
            resize_images = resize_test(images)
            outputs += grayAndNormlize_test(resize_images)
        return outputs

class Resize():
    def __init__(self, size):
        self.size = size

    def get_size(self, image_size):
        w, h = image_size
        if h <= self.size[1]:
            h = self.size[1]
        else:
            x = h % 32
            h = h - x
        if w < self.size[0]:
            w = self.size[0]
        else:
            y = w % 32
            w = w - y
        return (w, h)

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        output = []
        for image in images:
            size = self.get_size(image.size)
            image = image.resize(size)
            output.append(image)
        return output

class RandomCrop():
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self):
        size = random.randint(self.min_size, self.max_size)
        # h = w
        return (size, size)

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        w_crop, h_crop = self.get_size()
        w, h = images[0].size
        x = random.randint(0, w - w_crop)
        y = random.randint(0, h - h_crop)
        output = []
        for image in images:
            image = image.crop((x, y, x + w_crop, y + h_crop))
            output.append(image)
        return output

class GrayAndNormlize():
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, images, use_norm=True):
        if not isinstance(images, list):
            images = [images]
        output = []
        for image in images:
            image_color = np.array(image).astype(np.float32)
            image_gray = np.array(image.convert('L')).astype(np.float32)
            if use_norm:
                image_color[..., 0] += self.mean[0]
                image_color[..., 1] += self.mean[1]
                image_color[..., 2] += self.mean[2]
            image_color = torch.from_numpy(image_color.transpose((2, 0, 1)))
            image_gray = torch.from_numpy(np.asarray([image_gray, image_gray, image_gray]))
            output.append([image_color / 127.5 - 1.0, image_gray / 127.5 - 1.0])
        return output
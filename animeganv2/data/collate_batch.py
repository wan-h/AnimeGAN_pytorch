# coding: utf-8
# Author: wanhui0729@gmail.com

import torch
class ImageBatchCollator(object):

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def _prepare(self, images):
        '''
        准备训练数据
        '''
        if not images:
            return None
        color_images = [image[0] for image in images]
        gray_images = [image[1] for image in images]
        return [torch.stack(color_images), torch.stack(gray_images)]

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        real_images = transposed_batch[0]
        style_images = transposed_batch[1]
        smooth_images = transposed_batch[2]
        img_ids = transposed_batch[3]
        return self._prepare(real_images), self._prepare(style_images), self._prepare(smooth_images), img_ids

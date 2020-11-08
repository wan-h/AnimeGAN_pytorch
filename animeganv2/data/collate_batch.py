# coding: utf-8
# Author: wanhui0729@gmail.com

class ImageBatchCollator(object):

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        real_images = transposed_batch[0]
        style_images = transposed_batch[1]
        smooth_images = transposed_batch[2]
        img_ids = transposed_batch[3]
        return real_images, style_images, smooth_images, img_ids

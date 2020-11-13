# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
import random
import torch.utils.data

class AnimeGanDataset(torch.utils.data.Dataset):
    def __init__(self, dataDir, split, transforms=None):
        assert split in ('train', 'test'), "Please check split supported."
        self.transforms = transforms
        self.split = split
        dataFolder = os.path.join(dataDir, self.split)
        if self.split == 'train':
            real_path = os.path.join(dataFolder, 'real')
            style_path = os.path.join(dataFolder, 'style')
            smooth_path = os.path.join(dataFolder, 'smooth')
            # 初始化做smooth处理
            if not os.path.exists(smooth_path):
                self._gen_smooth(real_path, smooth_path)
            self.real = os.listdir(real_path)
            self.style = os.listdir(style_path)
            self.smooth_path = smooth_path
            self._init_real_producer()
            self._init_style_producer()
        else:
            real_path = os.path.join(dataFolder, 'real')
            self.real = os.listdir(real_path)

    def _gen_smooth(self, real_path, smooth_path):
        pass

    def _init_real_producer(self):
        self.real_producer = self.real.copy()
        random.shuffle(self.real_producer)

    def _init_style_producer(self):
        self.style_producer = self.style.copy()
        random.shuffle(self.style_producer)

    def _real_consumer(self):
        if len(self.real_producer) == 0:
            self._init_real_producer()
        real = self.real_producer.pop()
        return real

    def _style_consumer(self):
        if len(self.style_producer) == 0:
            self._init_style_producer()
        style = self.style_producer.pop()
        return style

    def __getitem__(self, index):
        if self.split == 'train':
            real, style = self._real_consumer(), self._style_consumer()
            # 同名
            smooth = os.path.join(self.smooth_path, os.path.basename(style))
            real = cv2.imread(real)
            style = cv2.imread(style)
            smooth = cv2.imread(smooth)
            if self.transforms:
                [real, style, smooth] = self.transforms([real, style, smooth])
            return real, style, smooth, index
        else:
            real = self.real[index]
            real = cv2.imread(real)
            if self.transforms:
                [real] = self.transforms([real])
            return real, None, None, index

    def __len__(self):
        if self.split == 'train':
            return max(len(self.real), len(self.style))
        else:
            return len(self.real)

    # def __iter__(self):
    #     self.iternum = self.__len__()
    #     return self
    #
    # def __next__(self):
    #     self.iternum -= 1
    #     if self.iternum < 0:
    #         raise StopIteration
    #     else:
    #         return self.__getitem__(self.iternum)
# coding: utf-8
# Author: wanhui0729@gmail.com

# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import random
import torch.utils.data
from PIL import Image

class AnimeGanDataset(torch.utils.data.Dataset):
    def __init__(self, dataDir, split, transforms=None):
        self.transforms = transforms
        dataFolder = os.path.join(dataDir, split)
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

    def _gen_smooth(self, real_path, smooth_path):
        pass

    def _init_real_producer(self):
        self.real_producer = self.real.copy()
        random.shuffle(self.real_producer)

    def _init_style_producer(self):
        self.style_producer = self.style.copy()
        random.shuffle(self.style_producer)

    def _consumer(self):
        if len(self.real_producer) == 0:
            self._init_real_producer()
        if len(self.style_producer) == 0:
            self._init_style_producer()
        real = self.real_producer.pop()
        style = self.style_producer.pop()
        return real, style

    def __getitem__(self, index):
        real, style = self._consumer()
        # 同名
        smooth = os.path.join(self.smooth_path, os.path.basename(style))
        real = Image.open(real).convert("RGB")
        style = Image.open(style).convert("RGB")
        smooth = Image.open(smooth).convert("RGB")
        if self.transforms:
            real, style, smooth = self.transforms(real, style, smooth)
        return real, style, smooth, index

    def __len__(self):
        return max(len(self.real), len(self.style))
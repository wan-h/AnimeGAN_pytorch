# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
import random
from tqdm import tqdm
import numpy as np
import torch.utils.data
from PIL import Image
from animegan.utils.comm import is_main_process, synchronize

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
            if not os.path.exists(smooth_path) and is_main_process():
                self._gen_smooth(style_path, smooth_path)
            synchronize()
            self.real = [os.path.join(real_path, name) for name in os.listdir(real_path)]
            self.style = [os.path.join(style_path, name) for name in os.listdir(style_path)]
            self.smooth_path = smooth_path
            self._init_real_producer()
            self._init_style_producer()
        else:
            real_path = os.path.join(dataFolder, 'real')
            self.real = [os.path.join(real_path, name) for name in os.listdir(real_path)]

    def _gen_smooth(self, style_path, smooth_path):
        os.makedirs(smooth_path)
        for image in tqdm(os.listdir(style_path)):
            image_path = os.path.join(style_path, image)
            bgr_img = cv2.imread(image_path)
            gray_img = cv2.imread(image_path, 0)

            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            gauss = cv2.getGaussianKernel(kernel_size, 0)
            gauss = gauss * gauss.transpose(1, 0)

            pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
            edges = cv2.Canny(gray_img, 100, 200)
            dilation = cv2.dilate(edges, kernel)
            gauss_img = np.copy(bgr_img)
            idx = np.where(dilation != 0)
            for i in range(np.sum(dilation != 0)):
                gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0],
                                gauss))
                gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1],
                                gauss))
                gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2],
                                gauss))

            cv2.imwrite(os.path.join(smooth_path, image), gauss_img)

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
            real = Image.open(real).convert("RGB")
            style = Image.open(style).convert("RGB")
            smooth = Image.open(smooth).convert("RGB")
            if self.transforms:
                [real, style, smooth] = self.transforms([real, style, smooth])
            return real, style, smooth, index
        else:
            real = self.real[index]
            real = Image.open(real).convert("RGB")
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
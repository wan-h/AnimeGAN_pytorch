# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
import numpy as np
from multiprocessing import Pool

def smooth(image_path, output_path, info):
    print(info)
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

    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), gauss_img)


if __name__ == '__main__':
    PROCESS_NUM = 12
    style_path = "/data/datasets/animegan/your_name/train/style"
    output_path = os.path.join(os.path.dirname(style_path), 'smooth')
    os.mkdir(output_path)

    p = Pool(PROCESS_NUM)

    images = os.listdir(style_path)
    total_images_num = len(images)

    for i, image in enumerate(images):
        info = "{} / {}".format(i, total_images_num)
        p.apply_async(smooth, args=(os.path.join(style_path, image), output_path, info))
    p.close()
    p.join()

# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
from tqdm import tqdm

def read_img(image_path):
    img = cv2.imread(image_path)
    assert len(img.shape) == 3
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()
    return B, G, R

def get_mean(images_path):
    images = os.listdir(images_path)
    image_num = len(images)
    B_total = 0
    G_total = 0
    R_total = 0
    for image in tqdm(images) :
        image_path = os.path.join(images_path, image)
        bgr = read_img(image_path)
        B_total += bgr[0]
        G_total += bgr[1]
        R_total += bgr[2]

    B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
    mean = (B_mean + G_mean + R_mean)/3

    return mean-B_mean, mean-G_mean, mean-R_mean

if __name__ == '__main__':
    images_path = "/data/datasets/animegan/your_name/train/style"
    B_mean, G_mean, R_mean = get_mean(images_path)
    print("B_mean: {}\nG_mean: {}\nR_mean: {}".format(B_mean, G_mean, R_mean))
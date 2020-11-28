# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import cv2
import torch
import logging
import datetime
import numpy as np
from tqdm import tqdm
from animeganv2.utils.tm import Timer
from animeganv2.modeling.utils import adjust_brightness_from_src_to_dst
from animeganv2.utils.comm import is_main_process, get_world_size, synchronize

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda")
    inference_timer = Timer()
    for i, batch in enumerate(tqdm(data_loader)):
        real_images, _, _, image_ids = batch
        # color images
        images = real_images[0]
        images = images.to(device)
        with torch.no_grad():
            inference_timer.tic()
            output = model(images)
            if device == cuda_device:
                torch.cuda.synchronize()
            inference_timer.toc()
            output = [(img.to(cpu_device), o.to(cpu_device)) for img, o in zip(images, output)]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict, inference_timer.total_time

def _save_prediction_images(predictions, output_folder, epoch):
    save_path = os.path.join(output_folder, str(epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    for img_id, (img, pred) in tqdm(predictions.items()):
        ori_img = (img.squeeze() + 1.) / 2 * 255
        ori_img = ori_img.permute(1, 2, 0).numpy().astype(np.uint8)
        fake_img = (pred.squeeze() + 1.) / 2 * 255
        fake_img = fake_img.permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{img_id}_c.jpg'), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))
        fake_img = adjust_brightness_from_src_to_dst(fake_img, ori_img)
        cv2.imwrite(os.path.join(save_path, f'{img_id}_a.jpg'), cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_path, f'{img_id}_b.jpg'), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))

class Evaluator(object):
    def __init__(self, data_loader, device="cuda", output_folder=None, logger_name=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.logger = logging.getLogger(logger_name + ".inference")
        self.output_folder = output_folder

    def do_inference(self, model, epoch):
        num_devices = get_world_size()
        dataset = self.data_loader.dataset
        self.logger.info("Start evaluation on {} dataset({} images).".format(dataset.__class__.__name__, len(dataset)))
        predictions, total_time = compute_on_dataset(model, self.data_loader, self.device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time_str = str(datetime.timedelta(seconds=total_time))
        self.logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        if self.output_folder:
            self.logger.info("Start save generated images on {} dataset({} images).".format(dataset.__class__.__name__, len(dataset)))
            _save_prediction_images(predictions, self.output_folder, epoch-1)

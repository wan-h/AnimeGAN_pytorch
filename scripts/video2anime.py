# coding: utf-8
# Author: wanhui0729@gmail.com

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from animeganv2.configs import cfg
from animeganv2.modeling.generator import build_generator
from animeganv2.data.transforms.build import build_transforms
from animeganv2.utils.model_serialization import load_state_dict
from animeganv2.modeling.utils import adjust_brightness_from_src_to_dst


def get_model(model_weight, device):
    model = build_generator(cfg)
    checkpoint = torch.load(model_weight, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("models").pop("generator"))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    video = args.video
    model_weight = cfg.MODEL.WEIGHT
    device = torch.device(cfg.MODEL.DEVICE)

    model = get_model(model_weight, device)
    model.eval()
    transform = build_transforms(cfg, False)

    videoCapture = cv2.VideoCapture(video)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    w, h = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (int(w - w % 32), int(h - h % 32))
    # size = (256, 256)
    # fourcc = int(videoCapture.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    frame_num = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    videoWriter = cv2.VideoWriter('./anime.avi', fourcc, fps, size)

    for _ in tqdm(range(frame_num)):
        success, frame = videoCapture.read()
        # frame = cv2.resize(frame, (256, 256))
        input = transform([frame])[0][0].unsqueeze(0)
        input = input.to(device)
        with torch.no_grad():
            pred = model(input).cpu()
        pred_img = (pred.squeeze() + 1.) / 2 * 255
        pred_img = pred_img.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
        pred_img = adjust_brightness_from_src_to_dst(pred_img, frame)
        videoWriter.write(cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
    videoCapture.release()
    videoWriter.release()

if __name__ == '__main__':
    main()
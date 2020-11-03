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
from PIL import Image
from animegan.configs import cfg
from animegan.modeling.generator import build_generator
from animegan.data.transforms.build import build_transforms
from animegan.utils.model_serialization import load_state_dict
from animegan.modeling.utils import adjust_brightness_from_src_to_dst


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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_num = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    videoWriter = cv2.VideoWriter('./anime.mp4', fourcc, fps, size)
    c = 1
    for _ in tqdm(range(frame_num)):
        c += 1
        if c > 21: break
        success, frame = videoCapture.read()
        frame = cv2.resize(frame, (1920, 1080))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(frame)
        input = transform([input])[0][0].unsqueeze(0)
        input = input.to(device)
        with torch.no_grad():
            pred = model(input).cpu()
        pred_img = (pred.squeeze() + 1.) / 2 * 255
        pred_img = pred_img.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
        pred_img = adjust_brightness_from_src_to_dst(pred_img, frame)
        video_frame = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        videoWriter.write(video_frame)
    videoCapture.release()
    videoWriter.release()

if __name__ == '__main__':
    main()
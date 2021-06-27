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
        "--video_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
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
    video_path = args.video_path
    output_path = args.output_path
    model_weight = cfg.MODEL.WEIGHT
    device = torch.device(cfg.MODEL.DEVICE)

    model = get_model(model_weight, device)
    model.eval()
    transform = build_transforms(cfg, False)

    videos = os.listdir(video_path)
    for video in tqdm(videos):
        try:
            videoPath = os.path.join(video_path, video)
            outputDir = os.path.join(output_path, video)
            if os.path.exists(outputDir):
                continue
            os.mkdir(outputDir)
            videoCapture = cv2.VideoCapture(videoPath)

            frame_num = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in tqdm(range(frame_num)):
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
                cv2.imwrite(os.path.join(outputDir, '{}.jpg'.format(i)), video_frame)
            videoCapture.release()
        except:
            continue

if __name__ == '__main__':
    main()
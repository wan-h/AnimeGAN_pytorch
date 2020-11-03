# coding: utf-8
# Author: wanhui0729@gmail.com

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import cv2
import torch
import argparse
import numpy as np
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
        required=True,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--image",
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

    image_path = args.image
    model_weight = cfg.MODEL.WEIGHT
    device = torch.device(cfg.MODEL.DEVICE)

    model = get_model(model_weight, device)
    model.eval()

    transform = build_transforms(cfg, False)

    image = cv2.imread(image_path)

    input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input = Image.fromarray(input)
    input = transform([input])[0][0].unsqueeze(0)
    input = input.to(device)
    with torch.no_grad():
        pred = model(input).cpu()
    pred_img = (pred.squeeze() + 1.) / 2 * 255
    pred_img = pred_img.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    pred_img = adjust_brightness_from_src_to_dst(pred_img, image)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("", pred_img)
    cv2.waitKey()
    image = cv2.resize(image, pred_img.shape[:-1][::-1])
    concat_img = np.concatenate((image, pred_img), 1)
    cv2.imwrite(f"anime_{os.path.basename(image_path)}", concat_img)

if __name__ == '__main__':
    main()
# coding: utf-8
# Author: wanhui0729@gmail.com

'''
gram 特征可视化
相同场景的gram特征距离较近,灰度图聚类更好,颜色收到一定颜色影响
同一张图片的gram特征受颜色的影响
'''

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import cv2
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from animeganv2.configs import cfg
from animeganv2.modeling.backbone import build_backbone
from animeganv2.data.transforms.build import build_transforms
from animeganv2.modeling.utils import gram, prepare_feature_extract


def get_model(device):
    model = build_backbone(cfg)
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
        "--imagesPath",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100
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

    imagesPath = args.imagesPath
    num = args.num
    device = torch.device(cfg.MODEL.DEVICE)
    model = get_model(device)
    model.eval()
    transform = build_transforms(cfg, False)

    writer = SummaryWriter(log_dir="gramEmbedding")
    mat_list = []
    label_image_list = []
    metadata_list = []
    count = 0
    for image in tqdm(os.listdir(imagesPath)):
        count += 1
        if count > num: break
        image_input_ori = cv2.imread(os.path.join(imagesPath, image))
        image_tarnsform = transform([image_input_ori])[0]
        image_input_gray = image_tarnsform[1].unsqueeze(0).to(device)
        image_input_color = image_tarnsform[0].unsqueeze(0).to(device)
        with torch.no_grad():
            # color
            backbone_feature_map = model(prepare_feature_extract(image_input_color))
            gram_feature_map = gram(backbone_feature_map)
            gram_feature = gram_feature_map.flatten()
            mat_list.append(gram_feature)
            image_input_ori_show = cv2.cvtColor(image_input_ori, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            label_image_list.append(torch.from_numpy(image_input_ori_show) / 255.0)
            metadata_list.append(torch.zeros([1]))

            #gray
            backbone_feature_map = model(prepare_feature_extract(image_input_gray))
            gram_feature_map = gram(backbone_feature_map)
            gram_feature = gram_feature_map.flatten()
            mat_list.append(gram_feature)
            image_input_ori_show = cv2.cvtColor(image_input_ori, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            label_image_list.append(torch.from_numpy(image_input_ori_show) / 255.0)
            metadata_list.append(torch.ones([1]))
    writer.add_embedding(
        mat=torch.stack(mat_list),
        metadata=metadata_list,
        label_img=torch.stack(label_image_list)
    )

if __name__ == '__main__':
    main()
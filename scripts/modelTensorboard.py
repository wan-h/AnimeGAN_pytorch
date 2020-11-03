# coding: utf-8
# Author: wanhui0729@gmail.com

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import torch
from animegan.configs import cfg
from animegan.modeling.build import build_model
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    tensorboard_dir = os.path.join(project_path, 'graph')
    writer_b = SummaryWriter(os.path.join(tensorboard_dir, 'backbone'))
    writer_g = SummaryWriter(os.path.join(tensorboard_dir, 'generator'))
    writer_d = SummaryWriter(os.path.join(tensorboard_dir, 'discriminator'))
    model_backbone, model_generator, model_discriminator = build_model(cfg)
    input_dummy = torch.ones((8, 3, 256, 256))
    writer_b.add_graph(model_backbone, input_dummy)
    writer_g.add_graph(model_generator, input_dummy)
    writer_d.add_graph(model_discriminator, input_dummy)
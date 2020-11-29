# coding: utf-8
# Author: wanhui0729@gmail.com

from torch import nn as nn
from torch.nn import functional as F

class Layer_Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:])
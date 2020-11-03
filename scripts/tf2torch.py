# coding: utf-8
# Author: wanhui0729@gmail.com

# tensorflow
# conv1_1 (3, 3, 3, 64) (64,)
# conv1_2 (3, 3, 64, 64) (64,)
# conv2_1 (3, 3, 64, 128) (128,)
# conv2_2 (3, 3, 128, 128) (128,)
# conv3_1 (3, 3, 128, 256) (256,)
# conv3_2 (3, 3, 256, 256) (256,)
# conv3_3 (3, 3, 256, 256) (256,)
# conv3_4 (3, 3, 256, 256) (256,)
# conv4_1 (3, 3, 256, 512) (512,)
# conv4_2 (3, 3, 512, 512) (512,)
# conv4_3 (3, 3, 512, 512) (512,)
# conv4_4 (3, 3, 512, 512) (512,)
# conv5_1 (3, 3, 512, 512) (512,)
# conv5_2 (3, 3, 512, 512) (512,)
# conv5_3 (3, 3, 512, 512) (512,)
# conv5_4 (3, 3, 512, 512) (512,)
# fc6 (25088, 4096) (4096,)
# fc7 (4096, 4096) (4096,)
# fc8 (4096, 1000) (1000,)

# pytorch
# features.0.weight torch.Size([64, 3, 3, 3])
# features.0.bias torch.Size([64])
# features.2.weight torch.Size([64, 64, 3, 3])
# features.2.bias torch.Size([64])
# features.5.weight torch.Size([128, 64, 3, 3])
# features.5.bias torch.Size([128])
# features.7.weight torch.Size([128, 128, 3, 3])
# features.7.bias torch.Size([128])
# features.10.weight torch.Size([256, 128, 3, 3])
# features.10.bias torch.Size([256])
# features.12.weight torch.Size([256, 256, 3, 3])
# features.12.bias torch.Size([256])
# features.14.weight torch.Size([256, 256, 3, 3])
# features.14.bias torch.Size([256])
# features.16.weight torch.Size([256, 256, 3, 3])
# features.16.bias torch.Size([256])
# features.19.weight torch.Size([512, 256, 3, 3])
# features.19.bias torch.Size([512])
# features.21.weight torch.Size([512, 512, 3, 3])
# features.21.bias torch.Size([512])
# features.23.weight torch.Size([512, 512, 3, 3])
# features.23.bias torch.Size([512])
# features.25.weight torch.Size([512, 512, 3, 3])
# features.25.bias torch.Size([512])
# features.28.weight torch.Size([512, 512, 3, 3])
# features.28.bias torch.Size([512])
# features.30.weight torch.Size([512, 512, 3, 3])
# features.30.bias torch.Size([512])
# features.32.weight torch.Size([512, 512, 3, 3])
# features.32.bias torch.Size([512])
# features.34.weight torch.Size([512, 512, 3, 3])
# features.34.bias torch.Size([512])
# classifier.0.weight torch.Size([4096, 25088])
# classifier.0.bias torch.Size([4096])
# classifier.3.weight torch.Size([4096, 4096])
# classifier.3.bias torch.Size([4096])
# classifier.6.weight torch.Size([1000, 4096])
# classifier.6.bias torch.Size([1000])

exchange_map = {
    "features.0.weight": ("conv1_1", 0),
    "features.0.bias": ("conv1_1", 1),
    "features.2.weight": ("conv1_2", 0),
    "features.2.bias": ("conv1_2", 1),
    "features.5.weight": ("conv2_1", 0),
    "features.5.bias": ("conv2_1", 1),
    "features.7.weight": ("conv2_2", 0),
    "features.7.bias": ("conv2_2", 1),
    "features.10.weight": ("conv3_1", 0),
    "features.10.bias": ("conv3_1", 1),
    "features.12.weight": ("conv3_2", 0),
    "features.12.bias": ("conv3_2", 1),
    "features.14.weight": ("conv3_3", 0),
    "features.14.bias": ("conv3_3", 1),
    "features.16.weight": ("conv3_4", 0),
    "features.16.bias": ("conv3_4", 1),
    "features.19.weight": ("conv4_1", 0),
    "features.19.bias": ("conv4_1", 1),
    "features.21.weight": ("conv4_2", 0),
    "features.21.bias": ("conv4_2", 1),
    "features.23.weight": ("conv4_3", 0),
    "features.23.bias": ("conv4_3", 1),
    "features.25.weight": ("conv4_4", 0),
    "features.25.bias": ("conv4_4", 1),
    "features.28.weight": ("conv5_1", 0),
    "features.28.bias": ("conv5_1", 1),
    "features.30.weight": ("conv5_2", 0),
    "features.30.bias": ("conv5_2", 1),
    "features.32.weight": ("conv5_3", 0),
    "features.32.bias": ("conv5_3", 1),
    "features.34.weight": ("conv5_4", 0),
    "features.34.bias": ("conv5_4", 1),
    "classifier.0.weight": ("fc6", 0),
    "classifier.0.bias": ("fc6", 1),
    "classifier.3.weight": ("fc7", 0),
    "classifier.3.bias": ("fc7", 1),
    "classifier.6.weight": ("fc8", 0),
    "classifier.6.bias": ("fc8", 1),
}

import torch
import numpy as np
from torchvision.models import vgg19

def get_tf_dict(path):
    state_dict = np.load(path, encoding='latin1', allow_pickle=True).item()
    return state_dict

def get_torch_dict():
    state_dict = vgg19(pretrained=True).state_dict()
    return state_dict

def exchange():
    tf_state_dict = get_tf_dict('/data/datasets/animegan/vgg19.npy')
    torch_state_dict = get_torch_dict()
    for k, v in torch_state_dict.items():
        tf_k, tf_ind = exchange_map[k]
        tf_map_v = tf_state_dict[tf_k][tf_ind]
        # check shape eq
        if 'weight' in k:
            if tf_map_v.ndim == 4:
                tf_map_v = np.ascontiguousarray(tf_map_v.transpose(3, 2, 0, 1))
            elif v.ndim == 2:
                tf_map_v = np.ascontiguousarray(tf_map_v.transpose(1, 0))
        assert tuple(v.shape) == tf_map_v.shape

        # exchange
        print("Exchane torch [{}] to tf [{}:{}]".format(k, tf_k, tf_ind))
        torch_state_dict[k] = torch.from_numpy(tf_map_v)
    torch.save(torch_state_dict, "vgg_tf_2_torch.pth")

if __name__ == '__main__':
    exchange()
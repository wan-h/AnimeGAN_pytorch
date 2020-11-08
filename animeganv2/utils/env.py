# coding: utf-8
# Author: wanhui0729@gmail.com

import sys
from torch.utils.collect_env import get_pretty_env_info
import PIL

SUPPORTED_DENY = ['win32']

def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)

def collect_env_info():
    if sys.platform.lower() in SUPPORTED_DENY:
        return "Warning: collect_env_info not supported on {}.".format(sys.platform.lower())
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str

# coding: utf-8
# Author: wanhui0729@gmail.com

# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import sys
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path)

import argparse
import torch
# see https://github.com/pytorch/pytorch/issues/973
# torch.multiprocessing.set_sharing_strategy('file_system')
from animeganv2.configs import cfg
from animeganv2.lib.trainer import train
from animeganv2.utils.env import collect_env_info
from animeganv2.utils.tm import generate_datetime_str
from animeganv2.utils.comm import get_rank, synchronize
from animeganv2.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="PyTorch Detector Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # 所有剩余的命令行参数都被收集到一个列表中 opts
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    # init distributed
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # 继续训练使用相同的目录结构
    if cfg.MODEL.WEIGHT and not cfg.MODEL.TRANSFER_LEARNING:
        weight_dir = cfg.MODEL.WEIGHT
        model_record_dir = os.path.dirname(weight_dir)
        output_dir = os.path.dirname(model_record_dir)
    else:
        output_dir = os.path.join(os.path.abspath(cfg.OUTPUT_DIR), generate_datetime_str(formate='%Y%m%d-%H%M%S'))
    if get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        synchronize()
    logger_name = "AnimeGan"
    logFile = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(name=logger_name, distributed_rank=get_rank(), logFile=logFile)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger_name, output_dir)

if __name__ == '__main__':
    main()
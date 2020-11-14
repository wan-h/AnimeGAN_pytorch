# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import torch
import logging
from animeganv2.engine.inference import Evaluator
from animeganv2.engine.trainer import do_train
from animeganv2.modeling.build import build_model
from animeganv2.utils.comm import synchronize, get_rank
from animeganv2.utils.checkpoint import ModelCheckpointer
from animeganv2.data.build import make_datasets, make_data_loader
from animeganv2.solver.build import make_lr_scheduler, make_optimizer

def train(cfg, local_rank, distributed, logger_name, output_dir):
    model_backbone, model_generator, model_discriminator = build_model(cfg)
    logger = logging.getLogger(logger_name)
    logger.info(f"model backbone:\n{model_backbone}")
    logger.info(f"model generator:\n{model_generator}")
    logger.info(f"model discriminator:\n{model_discriminator}")

    device = torch.device(cfg.MODEL.DEVICE)
    model_backbone.to(device)
    model_generator.to(device)
    model_discriminator.to(device)

    arguments = {}
    arguments["iteration"] = 0

    datasets, epoch_sizes = make_datasets(cfg, is_train=True)
    # train阶段dataset合并成一个
    epoch_size = epoch_sizes[0]

    optimizer_generator = make_optimizer(cfg, model_generator)
    optimizer_discriminator = make_optimizer(cfg, model_discriminator)
    # TODO: epoch_size优化
    scheduler_generator = make_lr_scheduler(cfg, optimizer_generator, epoch_size)
    scheduler_discriminator = make_lr_scheduler(cfg, optimizer_discriminator, epoch_size)

    if distributed:
        model_backbone = torch.nn.parallel.DistributedDataParallel(
            model_backbone, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True,
        )
        model_generator = torch.nn.parallel.DistributedDataParallel(
            model_generator, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True,
        )
        model_discriminator = torch.nn.parallel.DistributedDataParallel(
            model_discriminator, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True,
        )
        synchronize()
    models = {
        "generator": model_generator,
        "discriminator": model_discriminator
    }
    optimizers = {
        "generator": optimizer_generator,
        "discriminator": optimizer_discriminator
    }
    schedulers = {
        "generator": scheduler_generator,
        "discriminator": scheduler_discriminator
    }

    checkpointer = ModelCheckpointer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        save_dir=output_dir,
        logger_name=logger_name
    )

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.MODEL.TRANSFER_LEARNING)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg=cfg,
        datasets=datasets,
        epoch_sizes=epoch_sizes,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"]
    )

    evaluators = get_evaluator(cfg, distributed, logger_name, output_dir=output_dir)
    models.update({"backbone": model_backbone})

    do_train(
        models=models,
        cfg=cfg,
        data_loader=data_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        checkpointer=checkpointer,
        arguments=arguments,
        logger_name=logger_name,
        epoch_size=epoch_size,
        evaluators=evaluators,
    )

def get_evaluator(cfg, distributed, logger_name, dataEntrance=None, output_dir=None):
    torch.cuda.empty_cache()

    output_folders = list()
    datasetsInfo = cfg.DATASETS.TEST

    if output_dir:
        for datasetInfo in datasetsInfo:
            _output_folder = os.path.join(output_dir,
                                          "inference",
                                          datasetInfo.get('factory')+'_'+datasetInfo.get('split'))
            if get_rank() == 0:
                os.makedirs(_output_folder, exist_ok=True)
            output_folders.append(_output_folder)
    datasets_test, epoch_sizes = make_datasets(cfg, is_train=False, dataEntrance=dataEntrance)
    data_loaders_test = make_data_loader(cfg, datasets=datasets_test, epoch_sizes=epoch_sizes, is_train=False, is_distributed=distributed)
    evaluators = list()
    for output_folder, data_loader_test in zip(output_folders, data_loaders_test):
        evaluators.append(
            Evaluator(
                data_loader=data_loader_test,
                logger_name=logger_name,
                device=cfg.MODEL.DEVICE,
                output_folder=output_folder,
            )
        )
    return evaluators
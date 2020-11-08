# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import torch
import logging
from animeganv2.engine.trainer import do_train
from animeganv2.modeling.build import build_model
from animeganv2.utils.comm import synchronize, get_rank
from animeganv2.utils.checkpoint import ModelCheckpointer
from animeganv2.data.build import make_datasets, make_data_loader
from animeganv2.solver.build import make_lr_scheduler, make_optimizer

def train(cfg, local_rank, distributed, logger_name, output_dir, dataEntrance=None):
    model = build_model(cfg)

    logger = logging.getLogger(logger_name)
    logger.info(model)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    arguments = {}
    arguments["iteration"] = 0

    datasets, epoch_sizes = make_datasets(cfg, is_train=True, dataEntrance=dataEntrance)
    # train阶段dataset合并成一个
    epoch_size = epoch_sizes[0]

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, epoch_size)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True,
        )
        synchronize()

    checkpointer = ModelCheckpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
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

    evaluators = get_evaluator(cfg, distributed, logger_name, dataEntrance, output_dir)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print_period = cfg.SOLVER.PRINT_PERIOD
    test_period = cfg.SOLVER.TEST_PERIOD

    do_train(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer,
        device=device,
        checkpoint_period=checkpoint_period,
        print_period=print_period,
        test_period=test_period,
        arguments=arguments,
        logger_name=logger_name,
        epoch_size=epoch_size,
        evaluators=evaluators,
    )
    return model

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
                evaluate_type=cfg.TEST.TYPE
            )
        )
    return evaluators
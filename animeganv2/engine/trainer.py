# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import math
import logging
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from animeganv2.utils.tm import *
from animeganv2.utils.comm import synchronize
from animeganv2.utils.logger import MetricLogger
from animeganv2.utils.comm import get_world_size, is_main_process
from animeganv2.modeling.loss import *

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        loss_values = []
        # for k, v in loss_dict.items():
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            # all_losses.append(v)
            loss_values.append(loss_dict[k])
        loss_values = torch.stack(loss_values, dim=0)
        dist.reduce(loss_values, dst=0)
        if is_main_process():
            # only main process gets accumulated, so only divide by world_size in this case
            loss_values /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, loss_values)}
    return reduced_losses


def do_train(
    models,
    cfg,
    data_loader,
    optimizers,
    schedulers,
    checkpointer,
    arguments,
    logger_name,
    epoch_size,
    evaluators=None,
):
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print_period = cfg.SOLVER.PRINT_PERIOD
    test_period = cfg.SOLVER.TEST_PERIOD
    device = torch.device(cfg.MODEL.DEVICE)

    if is_main_process():
        save_dir = checkpointer.save_dir
        out_dir = os.path.dirname(save_dir)
        tensorboard_dir = os.path.join(out_dir, 'tensorboard')
        writer = SummaryWriter(tensorboard_dir)

    start_iter = arguments["iteration"]
    model_backbone = models['backbone']
    model_generator = models['generator']
    model_discriminator = models['discriminator']
    optimizer_generator = optimizers['generator']
    optimizer_discriminator = optimizers['discriminator']
    scheduler_generator = schedulers['generator']
    scheduler_discriminator = schedulers['discriminator']

    model_backbone.eval()
    model_generator.train()
    model_discriminator.train()

    logger = logging.getLogger(logger_name + ".trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)

    training_timer = Timer()
    training_timer.tic()
    batch_timer = Timer()
    batch_timer.tic()
    data_load_timer = Timer()
    data_load_timer.tic()
    _t = Timer()

    # epoch方式防止重复计算
    save_epoch = 0
    test_epoch = 0

    for iteration, (real_images, style_images, smooth_images, _) in enumerate(data_loader, start_iter):
        data_load_time = data_load_timer.toc()

        arguments["iteration"] = iteration
        # 当前epoch度量
        epoch_current = math.ceil((iteration + 1) / epoch_size)

        # init阶段
        if epoch_current < cfg.SOLVER.GENERATOR.INIT_EPOCH:
            # FP
            _t.tic()
            real_images = real_images.to(device)
            loss_init = init_loss(model_backbone, model_generator, real_images)
            FP_time = _t.toc()
            loss_dict = {"Init_loss": loss_init}
            if is_main_process():
                writer.add_scalars('train/loss', loss_dict, iteration)
            meters.update(**loss_dict)
            # BP
            optimizer_generator.zero_grad()
            loss_init.backward()
            optimizer_generator.step()
            scheduler_generator.step()
            scheduler_discriminator.step()
            BP_time = _t.toc()
            batch_time = batch_timer.toc()
        # 正常训练阶段
        else:
            # update D
            # FP
            if iteration % cfg.MODEL.COMMON.TRAINING_RATE == 0:
                pass

        # logger
        meters.update(batch_time=batch_time, data_time=data_load_time, FP_time=FP_time, BP_time=BP_time)
        eta_seconds = meters.batch_time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if ((iteration - 1) % print_period == 0) or iteration == max_iter - 1:
            epoch_string = "{epoch} | {epoch_iter:>4}/{epoch_size:<4}". \
                format(epoch=epoch_current, epoch_iter=iteration % epoch_size, epoch_size=epoch_size)

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        # "iter: {iter}",
                        "epoch: {epoch}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    # iter=iteration,
                    epoch=epoch_string,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0,
                )
            )

        # checkpointer
        if (epoch_current - 1) % checkpoint_period == 0 and epoch_current != save_epoch and epoch_current > 1:
            checkpointer.save("model_{:05d}".format(epoch_current - 1), **arguments)
            save_epoch = epoch_current
        if iteration == max_iter - 1:
            # checkpointer.save("model_final", **arguments)
            checkpointer.save("model_{:05d}".format(epoch_current), **arguments)

        # test
        if iteration == (max_iter - 1) or \
                ((epoch_current - 1) % test_period == 0 and epoch_current != test_epoch and epoch_current > 1):
            model.eval()
            # 最后一个epoch训练完成后做正确显示
            epoch_current_show = epoch_current if iteration == (max_iter - 1) else (epoch_current - 1)
            if evaluators:
                for evaluator in evaluators:
                    result = evaluator.do_inference(model)
                    # 只有主线程返回
                    if result:
                        # 用于解析日志标志
                        logger.info("(*^_^*)")
                        logger.info("Test model at {} dataset at {} epoch".
                                    format(evaluator.data_loader.dataset.__class__.__name__, epoch_current_show))
                        logger.info(result)
                        if is_main_process():
                            for k, v in result.items():
                                writer.add_scalar("test/{}".format(k), v, epoch_current)
                    # synchronize after test
                    synchronize()
            test_epoch = epoch_current
            model.train()

        batch_timer.tic()
        data_load_timer.tic()
    # checkpointer.save("model_final", **arguments)
    total_training_time = training_timer.toc()
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if is_main_process():
        writer.close()
    # synchronize after trainer
    synchronize()

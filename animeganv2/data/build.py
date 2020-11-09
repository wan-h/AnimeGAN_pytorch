# coding: utf-8
# Author: wanhui0729@gmail.com

# coding: utf-8
# Author: wanhui0729@gmail.com

import bisect
import copy
import torch.utils.data
from . import samplers
from . import datasets as dataInterface
from .collate_batch import ImageBatchCollator
from .transforms.build import build_transforms
from .datasets.concat_dataset import ConcatDataset
from animeganv2.utils.comm import get_world_size
from animeganv2.utils.datasetInfo import DatasetInfo

def build_dataset(dataInterface, datasetsInfo, transforms, is_train=True):
    if not isinstance(datasetsInfo, (list, tuple)):
        raise RuntimeError(
            "datasetsInfo should be a list"
        )
    datasets = []
    for datasetInfo in datasetsInfo:
        data = datasetInfo.get()
        factory = getattr(dataInterface, data.get("factory"))
        args = data.get("args")
        args["transforms"] = transforms
        # make dataset from factory
        try:
            dataset = factory(**args)
        except:
            raise ValueError("Please check dataset factory, the parameters are: {}".format(args))
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    '''
    Arguments:
        dataset: 数据集
    return:
        数据集中数据宽高比例
    '''
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, is_train=False
):
    # 防止batchsize为1时导致模型batchnorm报错
    drop = True if is_train else False
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=drop
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=drop
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_datasetsInfo(datasetsDictInfo):
    datasetsInfo = list()
    for datasetDictInfo in datasetsDictInfo:
        datasetsInfo.append(DatasetInfo(**datasetDictInfo))
    return datasetsInfo

def make_datasets(cfg, is_train=True):
    '''
    Arguments:
        cfg: 配置文件
        is_train: bool,是否处于训练阶段
    return:
        datasets: list of dataset
        epoch_sizes: list of size of the dataset's ecpoch
    '''
    datasetsInfo = make_datasetsInfo(cfg.DATASETS.TRAIN) if is_train \
        else make_datasetsInfo(cfg.DATASETS.TEST)

    transforms = build_transforms(cfg, is_train)
    dataEntrance = dataEntrance or dataInterface
    datasets = build_dataset(dataEntrance, datasetsInfo, transforms, is_train=is_train)

    epoch_sizes = []
    for dataset in datasets:
        epoch_size = len(dataset) // cfg.SOLVER.IMS_PER_BATCH
        epoch_sizes.append(epoch_size)
    if is_train:
        # during training, a single (possibly concatenated) datasets is returned
        assert len(datasets) == 1
    return datasets, epoch_sizes

def make_data_loader(cfg, datasets, epoch_sizes, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_EPOCH * epoch_sizes[0]
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        start_iter = 0
        num_iters = None

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, is_train
        )
        collator = ImageBatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

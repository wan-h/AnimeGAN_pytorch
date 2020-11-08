# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import torch
import logging
from animeganv2.utils.comm import get_rank
from animeganv2.utils.model_zoo import cache_url
from animeganv2.utils.model_serialization import load_state_dict


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir=None,
            logger_name="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if save_dir is not None:
            self.save_dir = os.path.join(save_dir, "model_record")
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, transfer_learning=False):
        '''
        Arguments:
            f: checkpoint file
            transfer_learning: bool, continue train if False
        return:
            checkpoint
        '''
        if not f:
            if self.has_checkpoint():
                # override argument with existing checkpoint
                f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))

        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

        if transfer_learning:
            checkpoint["iteration"] = 0
            return checkpoint

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        if self.save_dir is None:
            return False
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        assert self.save_dir is not None
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        assert self.save_dir is not None
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class ModelCheckpointer(Checkpointer):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir=None,
            logger_name=None,
    ):
        if get_rank() != 0:
            save_dir = None
        super(ModelCheckpointer, self).__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            logger_name=logger_name
        )

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            # print('cached_f:', cached_f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # load native detectron.pytorch checkpoint
        loaded = super(ModelCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

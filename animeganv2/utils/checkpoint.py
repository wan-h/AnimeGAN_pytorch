# coding: utf-8
# Author: wanhui0729@gmail.com

import os
import torch
import logging
from animeganv2.utils.comm import get_rank
from animeganv2.utils.model_serialization import load_state_dict

class ModelCheckpointer():
    def __init__(
            self,
            models,
            optimizers=None,
            schedulers=None,
            save_dir=None,
            logger_name=None,
    ):
        if get_rank() != 0:
            save_dir = None
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
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

        data = {"models": {}, "optimizers": {}, "schedulers": {}}
        data["models"]["generator"] = self.models['generator'].state_dict()
        data["models"]["discriminator"] = self.models['discriminator'].state_dict()
        if self.optimizers is not None:
            data["optimizers"]["generator"] = self.optimizers['generator'].state_dict()
            data["optimizers"]["discriminator"] = self.optimizers['discriminator'].state_dict()
        if self.schedulers is not None:
            data["schedulers"]["generator"] = self.schedulers['generator'].state_dict()
            data["schedulers"]["discriminator"] = self.schedulers['discriminator'].state_dict()

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

        if "optimizers" in checkpoint and self.optimizers:
            self.logger.info("Loading optimizer from {}".format(f))
            checkpoint_optimizers = checkpoint.pop("optimizers")
            self.optimizers["generator"].load_state_dict(checkpoint_optimizers.pop("generator"))
            self.optimizers["discriminator"].load_state_dict(checkpoint_optimizers.pop("discriminator"))
        if "schedulers" in checkpoint and self.schedulers:
            self.logger.info("Loading scheduler from {}".format(f))
            checkpoint_schedulers = checkpoint.pop("schedulers")
            self.schedulers["generator"].load_state_dict(checkpoint_schedulers.pop("generator"))
            self.schedulers["discriminator"].load_state_dict(checkpoint_schedulers.pop("discriminator"))
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
        checkpoint_models = checkpoint.pop("models")
        load_state_dict(self.models["generator"], checkpoint_models.pop("generator"))
        load_state_dict(self.models["discriminator"], checkpoint_models.pop("discriminator"))

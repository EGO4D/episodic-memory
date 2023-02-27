import logging
import os
import traceback

import detectron2.utils.comm as comm
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss
from torch.nn.parallel import (
    DataParallel,
    DistributedDataParallel,
    DistributedDataParallel as DDP,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tracking.dataset.build import build_dataloaders
from tracking.models.stark_tracker.stark_st import build_starkst
from tracking.models.stark_tracker.utils.box_ops import giou_loss
from tracking.solver.build import build_optimizer_scheduler
from tracking.utils.env import pathmgr

LOSS_FUNCTIONS = {"giou": giou_loss, "l1": l1_loss, "cls": BCEWithLogitsLoss()}


def build_loss_function(cfg):
    loss_funcs = {}
    loss_weights = {}
    assert len(cfg.TRAIN.LOSS_FUNCTIONS) == len(cfg.TRAIN.LOSS_WEIGHTS)
    for loss_name, loss_weight in zip(cfg.TRAIN.LOSS_FUNCTIONS, cfg.TRAIN.LOSS_WEIGHTS):
        loss_funcs[loss_name] = LOSS_FUNCTIONS[loss_name]
        loss_weights[loss_name] = loss_weight

    return loss_funcs, loss_weights


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, cfg):
        """
        args:
            cfg - config for the entire training
        """
        model = build_starkst(cfg)
        # if comm.get_world_size() == 1:
        #     device = torch.device(f"cuda:{comm.get_local_rank()}")
        #     model = model.to(device)
        # else:
        device = torch.device("cuda")
        model.to(device)

        if comm.get_world_size() > 1:
            model = DDP(
                model, device_ids=[comm.get_local_rank()], find_unused_parameters=True
            )

        local_rank = comm.get_local_rank()
        train_loader, val_loader = build_dataloaders(cfg, local_rank)
        optimizer, lr_scheduler = build_optimizer_scheduler(cfg, model)
        objective, loss_weight = build_loss_function(cfg)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.loaders = [train_loader, val_loader]
        self.epoch = 0
        self.global_step = 0
        self.stats = {}
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR
        self.log_path = os.path.join(self.output_dir, "log.txt")
        self.print_interval = cfg.TRAIN.PRINT_INTERVAL
        self.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
        self.prev_checkpoint_dir = getattr(cfg, "PREV_CHECKPOINT_DIR", None)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        pathmgr.mkdirs(self.output_dir)
        pathmgr.mkdirs(self.checkpoint_dir)

        if comm.is_main_process():
            self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
            pathmgr.mkdirs(self.tensorboard_dir)
            self.writer = SummaryWriter(self.tensorboard_dir)

        self.device = device
        self.objective = objective
        self.loss_weight = loss_weight

    def train(
        self,
        load_latest=False,
        load_previous_ckpt=False,
    ):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
        """
        max_epochs = self.cfg.TRAIN.EPOCH
        epoch = -1
        num_tries = 1
        for _i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    self.load_checkpoint(
                        checkpoint=self.prev_checkpoint_dir,
                        fields=["state_dict"],
                    )

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.scheduler_type != "cosine":
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    # only save the last 10 checkpoints
                    if comm.is_main_process() and (
                        epoch > (max_epochs - 10)
                        or epoch % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0
                    ):
                        self.save_checkpoint()
            except Exception:
                print("Training crashed at epoch {}".format(epoch))
                # self.epoch -= 1
                # load_latest = True
                print("Traceback for the error!")
                print(traceback.format_exc())
                print("Restarting training from last epoch ...")
                raise Exception

        print("Finished training!")

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            model = self.model.module
        else:
            model = self.model

        model_type = type(model).__name__
        state = {
            "epoch": self.epoch,
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "model_info": getattr(model, "info", None),
            "constructor": getattr(model, "constructor", None),
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats,
            "cfg": self.cfg,
        }

        # First save as a tmp file
        file_path = os.path.join(
            self.checkpoint_dir, "{}_ep{:04d}.pth.tar".format(model_type, self.epoch)
        )
        with pathmgr.open(file_path, "wb") as f:
            torch.save(state, f)

    def load_checkpoint(
        self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False
    ):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            model = self.model.module
        else:
            model = self.model

        model_type = type(model).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = pathmgr.ls(os.path.join(self.checkpoint_dir))
            checkpoint_list = [
                ckpt for ckpt in checkpoint_list if ckpt.split("_")[0] == model_type
            ]
            checkpoint_list = sorted(checkpoint_list)
            if checkpoint_list:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_list[-1])
            else:
                logging.error("No matching checkpoint file found")
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                "{}_ep{:04d}.pth.tar".format(model_type, checkpoint),
            )
        elif isinstance(checkpoint, str):
            # Load from a checkpoint path, usually for finetuning purpose
            if pathmgr.isfile(checkpoint):
                checkpoint_path = checkpoint
                fields = ["state_dict"]
            elif pathmgr.isdir(checkpoint):
                # Load from a checkpoint directory, usually training result from previous stage
                checkpoint_list = pathmgr.ls(checkpoint)
                checkpoint_list = [
                    ckpt for ckpt in checkpoint_list if ckpt.split("_")[0] == model_type
                ]
                checkpoint_list = sorted(checkpoint_list)
                if checkpoint_list:
                    checkpoint_path = os.path.join(checkpoint, checkpoint_list[-1])
                else:
                    logging.error("No matching checkpoint file found")
                    return
            else:
                raise NotImplementedError
        else:
            raise TypeError

        # Load network
        with pathmgr.open(checkpoint_path, "rb") as f:
            checkpoint_dict = torch.load(f, map_location="cpu")

        if "model_type" in checkpoint_dict:
            assert (
                model_type == checkpoint_dict["model_type"]
            ), "Model is not of correct type."

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ["cfg"]

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(
            ["lr_scheduler", "constructor", "model_type", "actor_type", "model_info"]
        )

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == "state_dict":
                logging.info(f"Loading state_dict from {checkpoint_path}")
                model.load_state_dict(checkpoint_dict[key])
            elif key == "optimizer":
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

            # Set the net info
            if (
                key == "constructor"
                and load_constructor
                and checkpoint_dict["constructor"] is not None
            ):
                model.constructor = checkpoint_dict["constructor"]
            if key == "model_info" and checkpoint_dict["model_info"] is not None:
                model.info = checkpoint_dict["model_info"]

            # Update the epoch in lr scheduler
            if key == "epoch":
                self.lr_scheduler.last_epoch = self.epoch
                # 2021.1.10 Update the epoch in data_samplers
                for loader in self.loaders:
                    if isinstance(loader.sampler, DistributedSampler):
                        loader.sampler.set_epoch(self.epoch)
            return True


class StatValue:
    def __init__(self):
        self.clear()

    def reset(self):
        self.val = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val):
        self.val = val
        self.history.append(self.val)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.clear()
        self.has_new_data = False

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def new_epoch(self):
        if self.count > 0:
            self.history.append(self.avg)
            self.reset()
            self.has_new_data = True
        else:
            self.has_new_data = False

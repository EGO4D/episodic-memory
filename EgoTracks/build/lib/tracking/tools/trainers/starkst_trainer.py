import logging
import time
import traceback
from collections import OrderedDict

import torch
from detectron2.utils import comm
from torch.utils.data.distributed import DistributedSampler
from tracking.models.stark_tracker.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
)
from tracking.models.stark_tracker.utils.merge import merge_template_search
from tracking.models.stark_tracker.utils.misc import NestedTensor
from tracking.tools.trainers.base_trainer import AverageMeter, BaseTrainer, StatValue
from tracking.utils.env import pathmgr


class STARKSTrainer(BaseTrainer):
    def __init__(self, cfg):
        """
        args:
            cfg: config_file
        """
        super().__init__(cfg)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

    def _set_default_settings(self):
        # Dict of all default values
        default = {"print_interval": 10, "print_stats": None, "description": ""}

        for param, default_value in default.items():
            if getattr(self, param, None) is None:
                setattr(self, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.model.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            # if self.move_data_to_gpu:
            data = data.to(self.device)

            data["epoch"] = self.epoch
            data["cfg"] = self.cfg
            # forward pass
            loss, stats = self.forward_and_compute_loss(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.TRAIN.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.TRAIN.GRAD_CLIP_NORM
                    )
                self.optimizer.step()

            # update statistics
            batch_size = data["template_images"].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            if comm.is_main_process():
                self._print_stats(i, loader, batch_size)

    def forward_and_compute_loss(self, data):
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data["search_anno"]  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, stats = self.compute_losses(out_dict, gt_bboxes[0])

        # Increment global step for TensorBoard logging
        self.global_step += 1

        return loss, stats

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.cfg.DATA.TEMPLATE.NUMBER):
            template_img_i = data["template_images"][i].view(
                -1, *data["template_images"].shape[2:]
            )  # (batch, 3, 128, 128)
            template_att_i = data["template_att"][i].view(
                -1, *data["template_att"].shape[2:]
            )  # (batch, 128, 128)
            feat_dict_list.append(
                self.model(
                    img=NestedTensor(template_img_i, template_att_i), mode="backbone"
                )
            )

        # process the search regions (t-th frame)
        search_img = data["search_images"].view(
            -1, *data["search_images"].shape[2:]
        )  # (batch, 3, 320, 320)
        search_att = data["search_att"].view(
            -1, *data["search_att"].shape[2:]
        )  # (batch, 320, 320)
        feat_dict_list.append(
            self.model(img=NestedTensor(search_img, search_att), mode="backbone")
        )

        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list)
        out_dict, _, _ = self.model(
            seq_dict=seq_dict,
            mode="transformer",
            run_box_head=run_box_head,
            run_cls_head=run_cls_head,
        )
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(
            -1, 4
        )  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :]
            .repeat((1, num_queries, 1))
            .view(-1, 4)
            .clamp(min=0.0, max=1.0)
        )  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective["giou"](
                pred_boxes_vec, gt_boxes_vec
            )  # (BN,4) (BN,4)
        except Exception:
            logging.error(traceback.format_exc())
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # Mean IoU
        mean_iou = iou.detach().mean()
        # compute l1 loss
        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight["giou"] * giou_loss + self.loss_weight["l1"] * l1_loss

        # Write losses and mIoU to Tensorboard
        if comm.is_main_process():
            self.writer.add_scalar("Loss/l1", l1_loss, global_step=self.global_step)
            self.writer.add_scalar("Loss/giou", giou_loss, global_step=self.global_step)
            self.writer.add_scalar("Loss/total", loss, global_step=self.global_step)
            self.writer.add_scalar("mean_iou", mean_iou, global_step=self.global_step)

        if return_status:
            # status for log
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "IoU": mean_iou.item(),
            }
            return loss, status
        else:
            return loss

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)
                if loader.name == "val":
                    comm.synchronize()

        self._stats_new_epoch()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict(
                {name: AverageMeter() for name in new_stats.keys()}
            )

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.print_interval == 0 or i == loader.__len__():
            print_str = "[%s: %d, %d / %d] " % (
                loader.name,
                self.epoch,
                i,
                loader.__len__(),
            )
            print_str += "FPS: %.1f (%.1f)  ,  " % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if self.print_stats is None or name in self.print_stats:
                    if hasattr(val, "avg"):
                        print_str += "%s: %.5f  ,  " % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            logging.error(print_str[:-5])
            with pathmgr.open(self.log_path, "a") as f:
                f.writelines(print_str[:-5] + "\n")

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except Exception:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = "LearningRate/group{}".format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, "new_epoch"):
                    stat_value.new_epoch()


class STARKSTTrainer(STARKSTrainer):
    def __init__(self, cfg):
        """
        args:
            cfg: config_file
        """
        super().__init__(cfg)

    def compute_losses(self, pred_dict, labels, return_status=True):
        loss = self.loss_weight["cls"] * self.objective["cls"](
            pred_dict["pred_logits"].view(-1), labels
        )

        # Write losses to Tensorboard
        if comm.is_main_process():
            self.writer.add_scalar("Loss/cls", loss, global_step=self.global_step)

        if return_status:
            # status for log
            status = {"cls_loss": loss.item()}
            return loss, status
        else:
            return loss

    def forward_and_compute_loss(self, data):
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth label
        labels = data["label"].view(-1)  # (batch, ) 0 or 1

        loss, status = self.compute_losses(out_dict, labels)

        # Increment global step for TensorBoard logging
        self.global_step += 1

        return loss, status

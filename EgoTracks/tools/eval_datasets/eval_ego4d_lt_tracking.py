import logging
import os
import pickle as pkl
import time

import detectron2.utils.comm as comm
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tracking.dataset.eval_datasets.ego4d_lt_tracking_dataset import (
    EGO4DLTTrackingDataset,
)
from tracking.metrics.lt_tracking_metrics import (
    compute_f_score,
    compute_precision_and_recall,
)
from tracking.metrics.miou import compute_overlaps
from tracking.tools.annotation.annotation_utils import seperate_occurrances
from tracking.utils.bbox_helper import xywh_2_cxywh
from tracking.utils.env import pathmgr
from tracking.utils.meters import AverageMeter, ProgressMeter
from tracking.utils.utils import opencv_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def eval_ego4d_lt_tracking(model, cfg):
    global_rank = comm.get_rank()
    cfg.defrost()
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "EGO4DLTTracking",
        f"{cfg.EVAL.EGO4DLT.TRACK_MODE}",
        f"{cfg.MODEL_TYPE}",
    )
    cfg.freeze()

    annotation_path = cfg.EVAL.EGO4DLT.ANNOTATION_PATH
    data_dir = cfg.EVAL.EGO4DLT.DATA_DIR
    track_mode = cfg.EVAL.EGO4DLT.TRACK_MODE
    use_visual_clip = cfg.EVAL.EGO4DLT.USE_VISUAL_CLIP
    eval_ratio = cfg.EVAL.EGO4DLT.EVAL_RATIO
    print_freq = cfg.EVAL.PRINT_FREQ

    # Initilize data loader
    logging.info("building dataset")
    eval_dataset = EGO4DLTTrackingDataset(
        data_dir, annotation_path, ratio=eval_ratio, split="test"
    )
    logging.info(f"{global_rank}: Length of eval dataset {len(eval_dataset)}")

    if comm.get_world_size() > 1:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False
        )
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            eval_sampler, 1, drop_last=False
        )
    else:
        batch_sampler = None
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=cfg.EVAL.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,  # don't batch, but yield individual elements
    )
    print(f"GPU {comm.get_local_rank()}: {len(eval_loader)}")
    # Setup stats logger
    result = {}
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, data_time],
    )

    save_dir = os.path.join(result_dir, "intermediate_result")
    pathmgr.mkdirs(save_dir)
    # switch to train mode
    model.eval()

    end = time.time()
    # currently batchsize = 1
    for i, data in enumerate(eval_loader):
        seq = data[0]
        seq_name = seq.name
        clip_uid, target_id = seq_name.split("_")[:2]
        object_title = "".join(seq_name.split("_")[2:])

        if use_visual_clip:
            raise NotImplementedError
        else:
            # visual_crop should only contain one element
            assert len(seq.visual_crop) == 1
            target_bbox = seq.visual_crop[list(seq.visual_crop.keys())[0]]
            target_frame_number = list(seq.visual_crop.keys())[0]

        logging.info(f"Processing {clip_uid}")

        if track_mode == "forward_backward_from_vcrop":
            total_frames = len(seq.frames)

            if cfg.EVAL.EGO4DLT.SAMPLE_5FPS == True:
                # This assumes the exported clips are 30FPS
                forward_frame_numbers = list(
                    range(target_frame_number, total_frames, 6)
                )
                backward_frame_numbers = list(range(target_frame_number, -1, -6))
            else:
                forward_frame_numbers = list(range(target_frame_number, total_frames))
                backward_frame_numbers = list(range(target_frame_number + 1))[::-1]

            forward_meta_data = {
                "target_bbox": target_bbox,
                "target_id": target_id,
                "frame_numbers": forward_frame_numbers,
            }
            backward_meta_data = {
                "target_bbox": target_bbox,
                "target_id": target_id,
                "frame_numbers": backward_frame_numbers,
            }

            pred_traj = model.inference(seq, forward_meta_data)
            forward_pred_bboxes = pred_traj[target_id]["bboxes"]
            model.reset_tracker()

            pred_traj = model.inference(seq, backward_meta_data)
            backward_pred_bboxes = pred_traj[target_id]["bboxes"]
            model.reset_tracker()

            pred_bboxes = backward_pred_bboxes[::-1][:-1] + forward_pred_bboxes
        else:
            raise NotImplementedError(f"Track mode {track_mode} is not implemented.")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        result[seq_name] = {
            "target_id": target_id,
            "object_title": object_title,
            "clip_uid": clip_uid,
            "seq_name": seq_name,
            "pred_bboxes": pred_bboxes,
        }
        model.reset_tracker()

        save_path = os.path.join(save_dir, f"{seq_name}.pkl")
        pkl.dump(result, pathmgr.open(save_path, "wb"))
        logging.info(f"Node {global_rank}. Saved to {save_path}")

    return result_dir


def gather_ego4d_lt_tracking_result(result):
    gathered = {}
    for seq_name, res in result.items():
        gathered[seq_name] = res

    return gathered

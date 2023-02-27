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
from tracking.dataset.eval_datasets.got10kdataset import GOT10KDataset
from tracking.metrics.miou import mIoU
from tracking.utils.env import pathmgr
from tracking.utils.meters import AverageMeter, ProgressMeter


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def eval_got10k(model, cfg):
    global_rank = comm.get_rank()
    cfg.defrost()
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "Got10k",
        "val",
    )
    cfg.freeze()

    root = cfg.DATA.GOT10K_DATA_DIR
    print_freq = cfg.EVAL.PRINT_FREQ

    # Initilize data loader
    logging.info("building dataset")
    eval_dataset = GOT10KDataset(root, split="val")
    logging.info(f"{global_rank}: Length of eval dataset {len(eval_dataset)}")

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, shuffle=False
    )
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        eval_sampler, 1, drop_last=False
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=1,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,  # don't batch, but yield individual elements
    )
    logging.info(f"{global_rank}: Length of eval dataloader {len(eval_loader)}")

    # Setup stats logger
    result = []
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, data_time],
    )

    # switch to train mode
    model.eval()

    end = time.time()
    # currently batchsize = 1
    for i, data in enumerate(eval_loader):
        seq = data[0]
        seq_name = seq.name
        target_id = seq_name
        gt_bboxes = seq.ground_truth_rect
        frame_numbers = list(range(len(gt_bboxes)))
        gt_frame_bbox_dict = {
            frame_number: gt_bboxes[i] for i, frame_number in enumerate(frame_numbers)
        }

        logging.info(f"Processing {seq_name}")

        meta_data = {
            "target_bbox": gt_frame_bbox_dict[frame_numbers[0]],
            "target_id": seq_name,
            "frame_numbers": frame_numbers,
        }

        pred_traj = model.inference_sequence(seq, meta_data)
        pred_bboxes = pred_traj[target_id]["bboxes"]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        result.append(
            {
                "target_id": target_id,
                "pred_bboxes": pred_bboxes,
                "gt_frame_bbox_dict": gt_frame_bbox_dict,
                "frame_numbers": frame_numbers,
            }
        )
        model.reset_tracker()

    save_dir = os.path.join(result_dir, "intermediate_result")
    pathmgr.mkdirs(save_dir)
    save_path = os.path.join(save_dir, f"{global_rank}.pkl")
    pkl.dump(result, pathmgr.open(save_path, "wb"))
    logging.info(f"Node {global_rank}. Saved to {save_path}")

    return result_dir


def calculate_got10k_metrics(cfg):
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "Got10k",
        "val",
    )
    intermediate_dir = os.path.join(result_dir, "intermediate_result")

    result = []
    # Strange, sometimes certain nodes do not save any result
    # for shard in range(total_machines):
    #     path = os.path.join(result_dir, "intermediate_result", f"{shard}.pkl")
    #     logging.info(path)
    #     shard_result = pkl.load(pathmgr.open(path, "rb"))
    #     result.extend(shard_result)
    files = pathmgr.ls(intermediate_dir)
    for f in files:
        path = os.path.join(intermediate_dir, f)
        logging.info(path)
        shard_result = pkl.load(pathmgr.open(path, "rb"))
        result.extend(shard_result)

    # result = gather_got10k(result)
    path = os.path.join(result_dir, "result.pkl")
    pkl.dump(result, pathmgr.open(path, "wb"))
    logging.info(f"Total number of predictions is {len(result)}.")

    iou_per_video = []
    all_pred_bboxes = []
    all_gt_bboxes = []

    for target_bboxes in result:
        # Only evaluate on frames annotated, all others ignored
        exclude_frame = [
            b["frame_number"] for b in target_bboxes["pred_bboxes"] if b["type"] == "gt"
        ]
        assert len(exclude_frame) == 1
        exclude_frame = exclude_frame[0]
        logging.info(exclude_frame)

        gt_frame_bbox_dict = target_bboxes["gt_frame_bbox_dict"]
        gt_frame_numbers = sorted(gt_frame_bbox_dict.keys())
        gt_frame_numbers = [
            frame for frame in gt_frame_numbers if frame != exclude_frame
        ]
        gt_frame_numbers_set = set(gt_frame_numbers)
        gt_bboxes = [gt_frame_bbox_dict[frame] for frame in gt_frame_numbers]

        pred_bboxes = [
            b
            for b in target_bboxes["pred_bboxes"]
            if b["frame_number"] in gt_frame_numbers_set
        ]
        pred_bboxes = [b["bbox"] for b in pred_bboxes]
        eval_result = mIoU(pred_bboxes, gt_bboxes)

        if len(pred_bboxes):
            eval_result = mIoU(pred_bboxes, gt_bboxes)
            all_pred_bboxes.extend(pred_bboxes)
            all_gt_bboxes.extend(gt_bboxes)
            iou_per_video.append(eval_result)

    average_bbox_iou = mIoU(all_pred_bboxes, all_gt_bboxes)
    mean_iou = np.mean(iou_per_video)
    logging.info(f"IoU per video: {iou_per_video}")
    logging.info(f"Eval result mIoU {mean_iou}, average bbox IoU {average_bbox_iou}.")

    return {
        "total_video": len(iou_per_video),
        "AO": mean_iou,
        "average bbox IoU": average_bbox_iou,
        "result_dir": result_dir,
        "MODEL_WEIGHTS": cfg.MODEL.WEIGHTS,
    }

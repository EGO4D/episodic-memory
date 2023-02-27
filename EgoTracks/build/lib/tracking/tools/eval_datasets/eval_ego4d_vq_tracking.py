import logging
import os
import pickle as pkl
import time
from collections import defaultdict

import detectron2.utils.comm as comm
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tracking.dataset.ego4d_tracking import Ego4DTracking
from tracking.metrics.lt_tracking_metrics import (
    compute_f_score,
    compute_precision_and_recall,
)
from tracking.metrics.miou import mIoU
from tracking.utils.env import pathmgr
from tracking.utils.meters import AverageMeter, ProgressMeter
from tracking.utils.utils import visualize_bbox


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def eval_ego4d_vq_tracking(model, cfg):
    global_rank = comm.get_rank()
    cfg.defrost()
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "EGO4DVQTracking",
        f"{cfg.EVAL.EGO4DVQ.TRACK_MODE}_{'5FPS' if cfg.EVAL.EGO4DVQ.IS_READ_5FPS_CLIP else '30FPS'}",
    )
    cfg.freeze()

    annotation_path = cfg.EVAL.EGO4DVQ.ANNOTATION_PATH
    clip_dir = cfg.EVAL.EGO4DVQ.CLIP_DIR
    video_dir = cfg.EVAL.EGO4DVQ.VIDEO_DIR
    is_read_5FPS_clip = cfg.EVAL.EGO4DVQ.IS_READ_5FPS_CLIP
    return_5FPS_frames = cfg.EVAL.EGO4DVQ.RETURN_5FPS_FRAMES
    visualize = cfg.EVAL.EGO4DVQ.VISUALIZE
    track_mode = cfg.EVAL.EGO4DVQ.TRACK_MODE
    print_freq = cfg.EVAL.PRINT_FREQ

    # Initilize data loader
    logging.info("building dataset")
    eval_dataset = Ego4DTracking(
        annotation_path,
        clip_dir,
        video_dir,
        is_read_5FPS_clip=is_read_5FPS_clip,
        return_5FPS_frames=return_5FPS_frames,
    )
    logging.info(f"{global_rank}: Length of eval dataset {len(eval_dataset)}")

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, shuffle=False
    )
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        eval_sampler, 1, drop_last=False
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=cfg.EVAL.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,  # don't batch, but yield individual elements
    )

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
        data = data[0]
        clip_uid = data["clip_uid"]
        object_title = data["object_title"]
        target_id = data["target_id"]
        imgs = data["imgs"]
        imgs_npy = imgs.permute(0, 2, 3, 1).contiguous().numpy()
        frame_bbox_dict = data["frame_bbox_dict"]
        object_title = data["object_title"]
        clip_uid = data["clip_uid"]
        frame_numbers = data["frame_numbers"]
        visual_crop = data["visual_crop"]

        logging.info(f"Processing {clip_uid}")

        if track_mode == "first_bbox":
            assert frame_numbers[0] in frame_bbox_dict
            meta_data = {
                "target_bbox": frame_bbox_dict[frame_numbers[0]],
                "target_id": target_id,
                "frame_numbers": frame_numbers,
            }

            pred_traj = model.inference(imgs, meta_data)
            pred_bboxes = pred_traj[target_id]["bboxes"]
        elif track_mode == "vq_bbox":
            vq_image = visual_crop["image"]
            vq_bbox = visual_crop["bbox"]
            vq_frame_number = visual_crop["frame_number"]

            # Concat the visual_crop to the first
            imgs = torch.cat((vq_image, imgs), dim=0)
            frame_numbers = [vq_frame_number] + frame_numbers
            imgs_npy = imgs.permute(0, 2, 3, 1).contiguous().numpy()

            meta_data = {
                "target_bbox": vq_bbox,
                "target_id": target_id,
                "frame_numbers": frame_numbers,
            }

            pred_traj = model.inference(imgs, meta_data)
            pred_bboxes = pred_traj[target_id]["bboxes"]
        elif track_mode == "largest_bbox":
            # calculate bounding box size for each frame
            frame_bbox_size_dict = {
                frame: bbox[2] * bbox[3] for frame, bbox in frame_bbox_dict.items()
            }
            frame_biggest_bbox = max(frame_bbox_size_dict, key=frame_bbox_size_dict.get)

            # get the array index of frame_numbers that correspond to the largest bbox
            index = np.where(np.array(frame_numbers) == frame_biggest_bbox)[0].item()
            assert frame_biggest_bbox == frame_numbers[index]

            target_bbox = frame_bbox_dict[frame_biggest_bbox]
            backward_frames = frame_numbers[: index + 1][::-1]
            # since imgs is a torch tensor, use flip to flip the video sequence
            backward_imgs = imgs[: index + 1].flip(0)

            forward_frames = frame_numbers[index:]
            forward_imgs = imgs[index:]

            forward_meta_data = {
                "target_bbox": target_bbox,
                "target_id": target_id,
                "frame_numbers": forward_frames,
            }
            backward_meta_data = {
                "target_bbox": target_bbox,
                "target_id": target_id,
                "frame_numbers": backward_frames,
            }

            pred_traj = model.inference(forward_imgs, forward_meta_data)
            forward_pred_bboxes = pred_traj[target_id]["bboxes"]
            model.reset_tracker()

            pred_traj = model.inference(backward_imgs, backward_meta_data)
            backward_pred_bboxes = pred_traj[target_id]["bboxes"]
            model.reset_tracker()

            # need to reverse the backward prediction since we reversed them before
            pred_bboxes = backward_pred_bboxes[::-1][:-1] + forward_pred_bboxes
        else:
            raise NotImplementedError(f"Track mode {track_mode} is not implemented.")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        result.append(
            {
                "target_id": target_id,
                "object_title": object_title,
                "clip_uid": clip_uid,
                "pred_bboxes": pred_bboxes,
                "gt_frame_bbox_dict": frame_bbox_dict,
                "frame_numbers": frame_numbers,
            }
        )
        model.reset_tracker()

        if visualize:
            visualized = visualize_bbox(
                imgs_npy.astype(np.uint8),
                [b["bbox"] for b in pred_bboxes],
                [b["score"] for b in pred_bboxes],
            )
            vis_dir = os.path.join(result_dir, "visualization")
            pathmgr.mkdirs(vis_dir)
            save_path = os.path.join(
                vis_dir, f"visualize_{clip_uid}_{target_id}_{object_title}.pkl"
            )
            pkl.dump(visualized, pathmgr.open(save_path, "wb"))

    save_dir = os.path.join(result_dir, "intermediate_result")
    pathmgr.mkdirs(save_dir)
    save_path = os.path.join(save_dir, f"{global_rank}.pkl")
    pkl.dump(result, pathmgr.open(save_path, "wb"))
    logging.info(f"Node {global_rank}. Saved to {save_path}")

    return result_dir


def gather_ego4d_vq_tracking_result(result):
    gathered = defaultdict(dict)
    for res in result:
        clip_uid = res["clip_uid"]
        target_id = res["target_id"]
        gathered[clip_uid][target_id] = res

    return gathered


def calculate_ego4d_vq_tracking_metrics(cfg):
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "EGO4DVQTracking",
        f"{cfg.EVAL.EGO4DVQ.TRACK_MODE}_{'5FPS' if cfg.EVAL.EGO4DVQ.IS_READ_5FPS_CLIP else '30FPS'}",
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

    result = gather_ego4d_vq_tracking_result(result)
    path = os.path.join(result_dir, "result.pkl")
    pkl.dump(result, pathmgr.open(path, "wb"))
    logging.info(f"Total number of predicted clips is {len(result)}.")

    iou_per_video = []
    all_pred_bboxes = []
    all_gt_bboxes = []
    all_pred_scores = []

    for _, clip_targets in result.items():
        for _, target_bboxes in clip_targets.items():
            # Only evaluate on frames annotated, all others ignored
            exclude_frame = [
                b["frame_number"]
                for b in target_bboxes["pred_bboxes"]
                if b["type"] == "gt"
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
            pred_scores = [b["score"] for b in pred_bboxes]
            pred_bboxes = [b["bbox"] for b in pred_bboxes]
            eval_result = mIoU(pred_bboxes, gt_bboxes)

            if len(pred_bboxes):
                eval_result = mIoU(pred_bboxes, gt_bboxes)
                all_pred_bboxes.append(pred_bboxes)
                all_gt_bboxes.append(gt_bboxes)
                all_pred_scores.append(pred_scores)
                iou_per_video.append(eval_result)

    average_bbox_iou = mIoU(
        [b for seq_bboxes in all_pred_bboxes for b in seq_bboxes],
        [b for seq_bboxes in all_gt_bboxes for b in seq_bboxes],
    )
    mean_iou = np.mean(iou_per_video)
    precision, recall = compute_precision_and_recall(
        all_pred_scores, all_pred_bboxes, all_gt_bboxes
    )
    f1_score, pr_score, re_score = compute_f_score(precision, recall)
    logging.info(f"IoU per video: {iou_per_video}")
    logging.info(
        f"Eval result mIoU {mean_iou}, average bbox IoU {average_bbox_iou}, f1 {f1_score}."
    )

    # save evaluation result
    path = os.path.join(result_dir, "eval_result.pkl")
    pkl.dump(
        {
            "total_video": len(iou_per_video),
            "F1": f1_score,
            "AO": mean_iou,
            "precision_score": pr_score,
            "re_score": re_score,
            "precision": precision,
            "recall": recall,
            "gt_frame_numbers": gt_frame_numbers,
            "average bbox IoU": average_bbox_iou,
            "result_dir": result_dir,
            "MODEL_WEIGHTS": cfg.MODEL.WEIGHTS,
        },
        pathmgr.open(path, "wb"),
    )

    return {
        "total_video": len(iou_per_video),
        "F1": f1_score,
        "precision_score": pr_score,
        "re_score": re_score,
        "AO": mean_iou,
        "average bbox IoU": average_bbox_iou,
        "result_dir": result_dir,
        "MODEL_WEIGHTS": cfg.MODEL.WEIGHTS,
    }

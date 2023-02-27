#!/usr/bin/env python3
"""
    Tests for the STARK tracker.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:stark_tracker_test -- --print-passing-details

"""
import json
import os
import unittest

import cv2
import torch
from tracking.dataset.ego4d_tracking import Ego4DTracking
from tracking.models.stark_tracker.params import parameters
from tracking.models.stark_tracker.stark_st import build_starkst
from tracking.models.stark_tracker.stark_tracker import STARKTracker
from tracking.utils.env import pathmgr, setup_environment
from tracking.utils.types import Params
from tracking.utils.utils import pad_bboxes, visualize_bbox


class TestSTARKTracker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSTARKTracker, self).__init__(*args, **kwargs)
        setup_environment()
        self.ANNOTATION_PATH = (
            "manifold://tracking/tree/ego4d/v1/annotations/vq_val.json"
        )
        self.CLIP_DIR = "manifold://tracking/tree/ego4d/clip"
        self.VIDEO_DIR = (
            "manifold://ego4d_fair/tree/intermediate/canonical/v7/full_scale/canonical"
        )

    def test_build_stark(self):
        params = parameters(None)
        checkpoint = params.checkpoint

        network = build_starkst(params.cfg)
        with pathmgr.open(checkpoint, "rb") as f:
            ckpt = torch.load(f)

        network.load_state_dict(ckpt["state_dict"], strict=True)

    def test_STARK_on_one_clip(self):
        # Define the tracker
        params = Params(
            model_path="manifold://tracking/tree/models/STARK/STARKST_ep0050.pth"
        )
        tracker = STARKTracker(params)

        dataset = Ego4DTracking(
            self.ANNOTATION_PATH, self.CLIP_DIR, self.VIDEO_DIR, is_read_5FPS_clip=True
        )
        data = dataset[100]

        imgs = data["imgs"]
        imgs_npy = imgs.permute(0, 2, 3, 1).contiguous().numpy()
        frame_bbox_dict = data["frame_bbox_dict"]
        target_id = data["target_id"]
        clip_uid = data["clip_uid"]
        frame_numbers = data["frame_numbers"]
        assert frame_numbers[0] in frame_bbox_dict

        bboxes = pad_bboxes(frame_bbox_dict, frame_numbers)
        meta_data = {
            "target_bbox": bboxes[0],
            "target_id": target_id,
            "frame_numbers": frame_numbers,
        }

        pred_traj = tracker.inference(imgs, meta_data)
        result = pred_traj[target_id]["bboxes"]
        pred_bboxes = []
        pred_scores = []
        for res in result:
            pred_bboxes.append(res["bbox"])
            pred_scores.append(res["score"])

        # Assert only return one object trajectory
        self.assertEqual(len(pred_traj), 1)
        # The # of bboxes should equal to # of scores
        self.assertEquals(len(pred_bboxes), len(pred_scores))
        # The # of predicted bboxes should be 1 less than the input sequences
        self.assertEquals(len(pred_bboxes), len(bboxes))

        # Visualize prediction to confirm
        # TODO Move below to tools/
        vis_imgs = visualize_bbox(imgs_npy, pred_bboxes)
        save_dir = f"/tmp/stark_tracker_test/{clip_uid}_{target_id}_pred"
        os.makedirs(save_dir, exist_ok=True)
        json.dump(pred_traj, open(os.path.join(save_dir, "pred.json"), "w"))

        for i, img in enumerate(vis_imgs):
            frame_number = frame_numbers[i]
            cv2.imwrite(os.path.join(save_dir, f"{frame_number}.png"), img)

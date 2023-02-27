#!/usr/bin/env python3
"""
    Tests for the evaluation_helper.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:ego4d_tracking_test

"""
import os
import unittest

import cv2
import numpy as np
import torch
from tracking.dataset.ego4d_tracking import Ego4DTracking
from tracking.utils.env import setup_environment
from tracking.utils.utils import pad_bboxes, visualize_bbox


class TestEgo4DTrackingDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEgo4DTrackingDataset, self).__init__(*args, **kwargs)
        setup_environment()
        self.ANNOTATION_PATH = (
            "manifold://tracking/tree/ego4d/v1/annotations/vq_val.json"
        )
        self.VIDEO_DIR = (
            "manifold://ego4d_fair/tree/intermediate/canonical/v7/full_scale/canonical"
        )
        self.CLIP_DIR = "manifold://tracking/tree/ego4d/clip"

    def test_ego4D_tracking_dataset_init(self):
        dataset = Ego4DTracking(self.ANNOTATION_PATH, self.CLIP_DIR, self.VIDEO_DIR)
        self.assertGreater(
            len(dataset), 10
        )  # check dataset is correctly loaded, 10 is a random number

    def test_ego4D_tracking_dataset_getitem(self):
        dataset = Ego4DTracking(self.ANNOTATION_PATH, self.CLIP_DIR, self.VIDEO_DIR)
        data = dataset[577]
        self.assertIn("imgs", data)
        self.assertIn("frame_bbox_dict", data)
        self.assertIn("frame_numbers", data)
        self.assertIn("clip_uid", data)
        self.assertIn("object_title", data)

        imgs = data["imgs"]
        frame_bbox_dict = data["frame_bbox_dict"]
        frame_numbers = data["frame_numbers"]
        bboxes = pad_bboxes(frame_bbox_dict, frame_numbers)
        self.assertEqual(
            len(imgs), len(bboxes)
        )  # each frame should have a bbounding box
        self.assertIsInstance(imgs, torch.Tensor)
        self.assertEqual(imgs.ndim, 4)  # [num_images, c, h, w]
        self.assertEqual(imgs.shape[1], 3)  # BGR three channels

    def test_ego4D_tracking_dataset_bbox_img_alignment(self):
        # TODO: Move below to tools/
        dataset = Ego4DTracking(
            self.ANNOTATION_PATH, self.CLIP_DIR, self.VIDEO_DIR, is_read_5FPS_clip=False
        )
        data = dataset[577]
        imgs = data["imgs"]
        frame_bbox_dict = data["frame_bbox_dict"]
        frame_numbers = data["frame_numbers"]
        clip_uid = data["clip_uid"]
        object_title = data["object_title"]

        # Visualize to confirm
        print(imgs.shape)
        imgs = imgs.permute(0, 2, 3, 1).contiguous()
        imgs = imgs.numpy().astype(np.uint8)
        bboxes = pad_bboxes(frame_bbox_dict, frame_numbers)
        vis_imgs = visualize_bbox(imgs, bboxes)
        save_dir = f"/tmp/Ego4DTracking_test/{clip_uid}_{object_title}"
        os.makedirs(save_dir, exist_ok=True)

        for i, img in enumerate(vis_imgs):
            frame_number = frame_numbers[i]
            cv2.imwrite(os.path.join(save_dir, f"{frame_number}.png"), img)

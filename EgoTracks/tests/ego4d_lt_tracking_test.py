#!/usr/bin/env python3
"""
    Tests for the evaluation_helper.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:ego4d_lt_tracking_test

"""
import os
import unittest

import cv2
from tracking.dataset.eval_datasets.base_dataset import Sequence
from tracking.dataset.eval_datasets.ego4d_lt_tracking_dataset import (
    EGO4DLTTrackingDataset,
)
from tracking.utils.env import pathmgr

from tracking.utils.utils import visualize_bbox


class TestEgo4DLTTrackingDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEgo4DLTTrackingDataset, self).__init__(*args, **kwargs)
        # The annotation is in the same format as Ego4D VQ.
        # We added two new fields to each query: lt_track and visual_clip
        # lt_track: contains bbox for each time we see the object
        # visual_clip: an extension of the visual crop in VQ. Visual crop only
        # contains a single template of the image, while visual clip extends this
        # to the entire occurance of this visual crop.
        self.ANNOTATION_PATH = (
            "manifold://tracking/tree/data/lt_tracking_annotation/lt_tracking_val.json"
        )
        self.DATA_DIR = "manifold://tracking/tree/data/ego4d/clips_frames"

    def test_ego4D_lt_tracking_dataset_init(self):
        dataset = EGO4DLTTrackingDataset(self.DATA_DIR, self.ANNOTATION_PATH)
        self.assertGreater(
            len(dataset), 2000
        )  # check dataset is correctly loaded, 2000 is a random number
        seq = dataset[0]
        self.assertIsInstance(seq, Sequence)

    def test_ego4D_tracking_dataset_bbox_img_alignment(self):
        # TODO: Move below to tools/
        dataset = EGO4DLTTrackingDataset(self.DATA_DIR, self.ANNOTATION_PATH)
        seq = dataset[100]
        clip_uid, target_id, object_title = seq.name.split("_")[:3]

        # Visualize to confirm
        visual_clip_frame_numbers = list(seq.visual_clip.keys())
        img_paths = [seq.frames[f] for f in visual_clip_frame_numbers]
        bboxes = [v for k, v in seq.visual_clip.items()]

        imgs = []
        for p in img_paths:
            p = pathmgr.get_local_path(p)
            img = cv2.imread(p)
            imgs.append(img)

        vis_imgs = visualize_bbox(imgs, bboxes)
        save_dir = f"/tmp/Ego4DTracking_test/{clip_uid}_{object_title}"
        os.makedirs(save_dir, exist_ok=True)

        for i, img in enumerate(vis_imgs):
            frame_number = visual_clip_frame_numbers[i]
            cv2.imwrite(os.path.join(save_dir, f"{frame_number}.png"), img)

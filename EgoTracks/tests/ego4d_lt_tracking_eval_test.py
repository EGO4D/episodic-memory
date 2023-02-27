#!/usr/bin/env python3
"""
    Tests for the evaluation_helper.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:ego4d_lt_tracking_eval_test

"""
import torch
from later.unittest import TestCase
from libfb.py.asyncio.unittest import async_test

from tracking.config.config import default_stark_config
from tracking.models.stark_tracker.stark_tracker import STARKTracker
from tracking.tools.eval_datasets.eval_ego4d_lt_tracking import (
    calculate_ego4d_lt_tracking_metrics,
    eval_ego4d_lt_tracking,
)


class TestEgo4DLTTrackingEval(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEgo4DLTTrackingEval, self).__init__(*args, **kwargs)
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
        self.config = default_stark_config
        # Make sure we only test on one sequence
        self.config.EVAL.EGO4DLT.EVAL_RATIO = 0.0001
        self.config.MODEL.WEIGHTS = (
            "manifold://tracking/tree/models/STARK/STARKST_ep0050.pth"
        )
        self.config.OUTPUT_DIR = (
            "manifold://tracking/tree/users/haotang/unit_test/ego4d_lttracking_eval"
        )
        self.config.TEST.UPDATE_INTERVALS.UPDATE_INTERVALS = [30]
        self.config.EVAL.EGO4DLT.PRE_DOWNLOAD = False

    @async_test
    async def test_ego4D_lt_tracking_eval_mode_first_visible(self):
        config = self.config.clone()
        config.EVAL.EGO4DLT.TRACK_MODE = "first_visible"
        model = STARKTracker(config, device=torch.device("cuda:0"))
        eval_ego4d_lt_tracking(model, config)
        res = calculate_ego4d_lt_tracking_metrics(config)
        self.assertIn("AO", res)
        self.assertIn("F1", res)

    @async_test
    async def test_ego4D_lt_tracking_eval_mode_first(self):
        config = self.config.clone()
        config.EVAL.EGO4DLT.TRACK_MODE = "first"
        model = STARKTracker(config, device=torch.device("cuda:0"))
        eval_ego4d_lt_tracking(model, config)
        res = calculate_ego4d_lt_tracking_metrics(config)
        self.assertIn("AO", res)
        self.assertIn("F1", res)

    @async_test
    async def test_ego4D_lt_tracking_eval_mode_forward_backward_from_vcrop(self):
        config = self.config.clone()
        config.EVAL.EGO4DLT.TRACK_MODE = "forward_backward_from_vcrop"
        model = STARKTracker(config, device=torch.device("cuda:0"))
        eval_ego4d_lt_tracking(model, config)
        res = calculate_ego4d_lt_tracking_metrics(config)
        self.assertIn("AO", res)
        self.assertIn("F1", res)

    @async_test
    async def test_ego4D_lt_tracking_eval_mode_occurrence(self):
        config = self.config.clone()
        config.EVAL.EGO4DLT.TRACK_MODE = "occurrence"
        model = STARKTracker(config, device=torch.device("cuda:0"))
        eval_ego4d_lt_tracking(model, config)
        res = calculate_ego4d_lt_tracking_metrics(config)
        self.assertIn("AO", res)
        self.assertIn("F1", res)

    @async_test
    async def test_ego4D_lt_tracking_STARK_R101_eval_mode_first(self):
        config = self.config.clone()
        config.EVAL.EGO4DLT.TRACK_MODE = "first"
        config.MODEL.WEIGHTS = (
            "manifold://tracking/tree/models/STARK/STARKST_Res101_ep0050.pth"
        )
        config.MODEL.BACKBONE.TYPE = "resnet101"
        model = STARKTracker(config, device=torch.device("cuda:0"))
        eval_ego4d_lt_tracking(model, config)
        res = calculate_ego4d_lt_tracking_metrics(config)
        self.assertIn("AO", res)
        self.assertIn("F1", res)

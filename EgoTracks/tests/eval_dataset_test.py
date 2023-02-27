#!/usr/bin/env python3
"""
    Tests for the tracking eval dataset.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:eval_dataset_test

"""
from later import unittest
from libfb.py.asyncio.unittest import async_test
from tracking.dataset.eval_datasets.got10kdataset import GOT10KDataset


class TestEgo4DTrackingDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEgo4DTrackingDataset, self).__init__(*args, **kwargs)

    @async_test
    async def test_GOT10K_dataloader(self):
        root = "manifold://tracking/tree/data/got10k"
        dataset = GOT10KDataset(root, split="val")
        data = dataset[0]
        self.assertIn("imgs", data)
        self.assertIn("gt_bboxes", data)
        self.assertIn("seq_name", data)

        imgs = data["imgs"]
        gt_bboxes = data["gt_bboxes"]
        self.assertEquals(len(imgs), len(gt_bboxes))

        # # visualize to verify manually
        # import numpy as np
        # import cv2
        # import os
        # from tracking.utils.utils import visualize_bbox

        # seq_name = data["seq_name"]
        # imgs = imgs.permute(0, 2, 3, 1).contiguous()
        # imgs = imgs.numpy().astype(np.uint8)
        # vis_imgs = visualize_bbox(imgs, gt_bboxes)
        # save_dir = f"/tmp/eval_dataset_test_got10k/{seq_name}"
        # os.makedirs(save_dir, exist_ok=True)

        # for i, img in enumerate(vis_imgs):
        #     cv2.imwrite(os.path.join(save_dir, f"{i}.png"), img)

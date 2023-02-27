#!/usr/bin/env python3
"""
    Tests for the STARK dataset.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:dataset_test

"""
import time

from later import unittest
from libfb.py.asyncio.unittest import async_test
from tracking.dataset.build import build_dataloaders
from tracking.utils.types import Params


class TestEgo4DTrackingDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEgo4DTrackingDataset, self).__init__(*args, **kwargs)

    @async_test
    async def test_GOT10K_dataloader(self):
        params = Params()
        params.STARK.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain"]
        params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
        params.STARK.DATA.DATA_FRACTION = None  # Faster test
        s = time.time()
        train_dataloader, val_dataloader = build_dataloaders(
            params.STARK, local_rank=-1
        )
        print(
            f"Loading {params.STARK.DATA.DATA_FRACTION} dataset takes {time.time() - s}"
        )
        s = time.time()
        data = train_dataloader.dataset[0]
        print(f"Load one data point takes {time.time() - s}")
        self.assertIn("template_images", data)
        self.assertIn("search_images", data)
        self.assertIn("template_anno", data)
        self.assertIn("search_anno", data)

    @async_test
    async def test_LASOT_dataloader(self):
        params = Params()
        params.STARK.DATA.TRAIN.DATASETS_NAME = ["LASOT"]
        params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
        params.STARK.DATA.DATA_FRACTION = None  # Faster test
        s = time.time()
        train_dataloader, val_dataloader = build_dataloaders(
            params.STARK, local_rank=-1
        )
        print(
            f"Loading {params.STARK.DATA.DATA_FRACTION} dataset takes {time.time() - s}"
        )
        s = time.time()
        data = train_dataloader.dataset[0]
        print(f"Load one data point takes {time.time() - s}")
        self.assertIn("template_images", data)
        self.assertIn("search_images", data)
        self.assertIn("template_anno", data)
        self.assertIn("search_anno", data)

    @async_test
    async def test_COCO_dataloader(self):
        params = Params()
        params.STARK.DATA.TRAIN.DATASETS_NAME = ["COCO17"]
        params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
        params.STARK.DATA.DATA_FRACTION = None  # Faster test
        s = time.time()
        train_dataloader, val_dataloader = build_dataloaders(
            params.STARK, local_rank=-1
        )
        print(
            f"Loading {params.STARK.DATA.DATA_FRACTION} dataset takes {time.time() - s}"
        )
        s = time.time()
        data = train_dataloader.dataset[0]
        print(f"Load one data point takes {time.time() - s}")
        self.assertIn("template_images", data)
        self.assertIn("search_images", data)
        self.assertIn("template_anno", data)
        self.assertIn("search_anno", data)

    @async_test
    async def test_EGO4DVQ_dataloader(self):
        params = Params()
        params.STARK.DATA.TRAIN.DATASETS_NAME = ["EGO4DVQ"]
        params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
        params.STARK.DATA.DATA_FRACTION = None  # Faster test
        s = time.time()
        train_dataloader, val_dataloader = build_dataloaders(
            params.STARK, local_rank=-1
        )
        print(
            f"Loading {params.STARK.DATA.DATA_FRACTION} dataset takes {time.time() - s}"
        )
        s = time.time()
        data = train_dataloader.dataset[0]
        print(f"Load one data point takes {time.time() - s}")
        self.assertIn("template_images", data)
        self.assertIn("search_images", data)
        self.assertIn("template_anno", data)
        self.assertIn("search_anno", data)

    async def test_TRACKINGNET_dataloader(self):
        params = Params()
        params.STARK.DATA.TRAIN.DATASETS_NAME = ["TRACKINGNET"]
        params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
        params.STARK.DATA.DATA_FRACTION = None  # Faster test
        s = time.time()
        train_dataloader, val_dataloader = build_dataloaders(
            params.STARK, local_rank=-1
        )
        print(
            f"Loading {params.STARK.DATA.DATA_FRACTION} dataset takes {time.time() - s}"
        )
        s = time.time()
        data = train_dataloader.dataset[0]
        print(f"Load one data point takes {time.time() - s}")
        self.assertIn("template_images", data)
        self.assertIn("search_images", data)
        self.assertIn("template_anno", data)
        self.assertIn("search_anno", data)

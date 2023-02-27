import os

import numpy as np
import torch
from tracking.utils.env import pathmgr
from tracking.utils.load_text import load_text

from .base_dataset import BaseDataset, Sequence, SequenceList


class _GOT10KDataset(BaseDataset):
    """GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root, split=None):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == "test" or split == "val":
            self.base_path = os.path.join(root, split)
        else:
            self.base_path = os.path.join(root, "train")

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = "{}/{}/groundtruth.txt".format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64)

        frames_path = "{}/{}".format(self.base_path, sequence_name)
        frame_list = [
            frame for frame in pathmgr.ls(frames_path) if frame.endswith(".jpg")
        ]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(
            sequence_name, frames_list, "got10k", ground_truth_rect.reshape(-1, 4)
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with pathmgr.open("{}/list.txt".format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == "ltrval":
            with open(
                "{}/got10k_val_split.txt".format(self.env_settings.dataspec_path)
            ) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list


class GOT10KDataset(torch.utils.data.Dataset):
    """GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root, split=None):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        dataset = _GOT10KDataset(root, split=split)
        self.sequences = dataset.get_sequence_list()
        self.sequence_list = dataset.sequence_list

    def __getitem__(self, index):
        # seq = self.sequences[index]
        # seq_name = self.sequence_list[index]
        # gt_bboxes = seq.ground_truth_rect
        # frame_paths = seq.frames

        # imgs = []
        # for path in frame_paths:
        #     image = opencv_loader(path)
        #     imgs.append(image)

        # try:
        #     imgs = np.array(imgs, dtype=np.float32)
        #     imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        # except Exception as e:
        #     import logging

        #     logging.info(f"{seq_name}")
        #     logging.info(e)
        #     raise RuntimeError

        # return {"imgs": imgs, "gt_bboxes": gt_bboxes, "seq_name": seq_name}
        return self.sequences[index]

    def __len__(self):
        return len(self.sequence_list)

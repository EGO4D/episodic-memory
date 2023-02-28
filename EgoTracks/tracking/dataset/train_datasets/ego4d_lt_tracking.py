import os
import os.path
import random
from itertools import groupby

import torch
from tracking.dataset.base_video_dataset import BaseVideoDataset
from tracking.dataset.eval_datasets.ego4d_lt_tracking_dataset import (
    EGO4DLTTrackingDataset,
)
from tracking.utils.utils import opencv_loader


class EGO4DLTT(BaseVideoDataset):
    """Ego4D VQ response track dataset."""

    def __init__(
        self,
        data_dir: str,
        annotation_path: str,
        data_fraction=None,
        image_loader=opencv_loader,
    ):
        """
        args:
            data_dir - directory that contains image files
            annotation_path - which annotation file to read
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        super().__init__("Ego4DLTT", annotation_path)
        self.ego4d_lt_tracking = EGO4DLTTrackingDataset(
            data_dir,
            annotation_path,
            ratio=data_fraction if data_fraction is not None else 1.0,
        )
        self.sequence_list = self.ego4d_lt_tracking.sequences

    def get_sequence_info(self, seq_id):
        bbox, frame_numbers = self._get_bbox_from_lt_tracking(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ones(len(bbox))

        return {
            "bbox": bbox,
            "valid": valid,
            "visible": visible,
            "frame_numbers": frame_numbers,
        }

    def get_name(self):
        return "ego4d_lt_tracking"

    def _get_bbox_from_lt_tracking(self, seq_id):
        frame_bbox_dict = self.ego4d_lt_tracking[seq_id].gt_bbox_dict
        frame_numbers = list(frame_bbox_dict.keys())

        bboxes = [frame_bbox_dict[frame_number] for frame_number in frame_numbers]
        bboxes = torch.tensor(bboxes)
        frame_numbers = torch.tensor(frame_numbers)

        return bboxes, frame_numbers

    def _get_frame(self, frame_path):
        return self.image_loader(frame_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq = self.ego4d_lt_tracking[seq_id]
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        obj_meta = {}
        frame_numbers = anno["frame_numbers"]

        frame_list = [
            self._get_frame(seq.frames[frame_numbers[f_id]]) for f_id in frame_ids
        ]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

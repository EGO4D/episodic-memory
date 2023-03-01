import os
import os.path
import random
from itertools import groupby

import torch
from tracking.dataset.base_video_dataset import BaseVideoDataset
from tracking.dataset.ego4d_tracking import Ego4DTracking
from tracking.utils.utils import opencv_loader


class Ego4DVQ(BaseVideoDataset):
    """Ego4D VQ response track dataset."""

    def __init__(
        self,
        root: str,
        annotation_path: str,
        data_fraction=None,
        image_loader=opencv_loader,
    ):
        """
        args:
            root - directory that contains image files
            annotation_path - which annotation file to read
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        super().__init__("Ego4DVQ", root)
        self.ego4d_tracking = Ego4DTracking(annotation_path, None, None)

        if data_fraction is not None:
            self.ego4d_tracking.response_tracks = random.sample(
                self.ego4d_tracking.response_tracks,
                int(len(self.ego4d_tracking.response_tracks) * data_fraction),
            )
        self.sequence_list = self.ego4d_tracking.response_tracks

    def get_sequence_info(self, seq_id):
        bbox, frame_numbers = self._get_bbox_from_response_track(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ones(len(bbox))

        return {
            "bbox": bbox,
            "valid": valid,
            "visible": visible,
            "frame_numbers": frame_numbers,
        }

    def get_name(self):
        return "ego4d_vq"

    def _get_sequence_path(self, seq_id):
        response_track = self.ego4d_tracking.response_tracks[seq_id]
        clip_uid, object_title, target_id = (
            response_track["clip_uid"],
            response_track["object_title"],
            response_track["target_id"],
        )
        seq_dir = f"{clip_uid}_{target_id}_{object_title}"

        return os.path.join(self.root, seq_dir)

    def _get_bbox_from_response_track(self, seq_id):
        response_track = self.ego4d_tracking.response_tracks[seq_id]["response_track"]
        frame_bbox_dict = {}
        frame_numbers = []
        for frame_number, data in groupby(response_track, lambda x: x["frame"]):
            data = list(data)
            assert len(data) == 1

            for b in data:
                box = [b["x"], b["y"], b["width"], b["height"]]
                frame_bbox_dict[frame_number] = box
                frame_numbers.append(frame_number)

        bboxes = [frame_bbox_dict[frame_number] for frame_number in frame_numbers]
        bboxes = torch.tensor(bboxes)
        frame_numbers = torch.tensor(frame_numbers)

        return bboxes, frame_numbers

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(
            seq_path, "{:08}.png".format(frame_id)
        )  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        obj_meta = {}
        frame_numbers = anno["frame_numbers"]

        frame_list = [
            self._get_frame(seq_path, frame_numbers[f_id]) for f_id in frame_ids
        ]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

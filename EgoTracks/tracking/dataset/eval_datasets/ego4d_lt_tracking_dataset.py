import json
import logging
import os
import time
from typing import Dict

import torch
from tqdm import tqdm
from tracking.utils.env import pathmgr

from .base_dataset import BaseDataset, Sequence, SequenceList


class _EGO4DLTTrackingDataset(BaseDataset):
    """EGO4D Long-term Tracking dataset."""

    def __init__(self, data_dir, annotation_path, split=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.frames_dir = os.path.join(data_dir, "frames")
        self.info_dir = os.path.join(data_dir, "clip_info")
        self.annotation_path = annotation_path

        self.clip_info = self.load_clip_info(os.path.join(data_dir, "clip_info.json"))
        self.sequences = self.get_sequences()
        self.sequence_list = [seq.name for seq in self.sequences]

    def get_sequences(self):
        with pathmgr.open(self.annotation_path, "r") as f:
            vq_annotations = json.load(f)
            vq_ann_video_uids = [x["video_uid"] for x in vq_annotations["videos"]]

        self.vq_video_dict = {
            x["video_uid"]: x["clips"] for x in vq_annotations["videos"]
        }

        # Load annotation all the visual query, including response track and visual crop.
        sequences = []
        for video_uid in tqdm(vq_ann_video_uids, total=len(vq_ann_video_uids)):
            clips = self.vq_video_dict[video_uid]
            for clip in clips:
                clip_uid = clip["exported_clip_uid"]

                response_tracks = self.get_lt_track(
                    video_uid, clip_uid, frame_number_key="exported_clip_frame_number"
                )
                sequences.extend(response_tracks)

        return SequenceList(sequences)

    def load_clip_info(self, clip_info_path):
        with pathmgr.open(clip_info_path) as f:
            clip_info = json.load(f)

        return clip_info

    def get_lt_track(
        self,
        video_uid: str,
        clip_uid: str,
        frame_number_key: str = "exported_clip_frame_number",
    ) -> Dict:
        vq_video_dict = self.vq_video_dict
        ann = vq_video_dict.get(video_uid)
        sequences = []

        for clip in ann:
            if not clip["exported_clip_uid"] == clip_uid:
                continue

            # Because we merge multiple annotations for each clip,
            # there should be only one annotation for each clip and
            # potentially more than three query sets
            assert len(clip["annotations"]) == 1
            for cann in clip["annotations"]:
                for target_id, query_set in cann["query_sets"].items():
                    object_title = query_set["object_title"]
                    # visual_crop
                    visual_crop = query_set["visual_crop"]
                    visual_crop = {
                        visual_crop[frame_number_key]: [
                            visual_crop["x"],
                            visual_crop["y"],
                            visual_crop["width"],
                            visual_crop["height"],
                        ]
                    }

                    # Cannot find the extracted frame dir
                    frames = self.clip_info[clip_uid]["frames"]
                    frames = [
                        os.path.join(self.frames_dir, clip_uid, f) for f in frames
                    ]
                    if frames is None:
                        continue

                    # if it's an unannotated test set, there are no these fields:
                    if self.split != "test":
                        if not query_set["is_valid"]:
                            continue
                        if "lt_track" not in query_set:
                            continue
                        visual_clip = {}
                        for frame in query_set["visual_clip"]:
                            visual_clip[frame[frame_number_key]] = [
                                frame["x"],
                                frame["y"],
                                frame["width"],
                                frame["height"],
                            ]
                        gt_bbox_dict = {}
                        for frame in query_set["lt_track"]:
                            gt_bbox_dict[frame[frame_number_key]] = [
                                frame["x"],
                                frame["y"],
                                frame["width"],
                                frame["height"],
                            ]
                        # Make sure the gt_bbox_dict contain all the previous bbox
                        gt_bbox_dict.update(visual_clip)
                        gt_bbox_dict.update(visual_crop)
                    else:
                        visual_clip = None
                        gt_bbox_dict = {}

                    seq = Sequence(
                        f"{clip_uid}_{target_id}_{object_title}",
                        frames,
                        "ego4d_lt_tracking",
                        None,
                        object_ids=[f"{target_id}_{object_title}"],
                    )
                    seq.visual_crop = visual_crop
                    seq.visual_clip = visual_clip
                    seq.gt_bbox_dict = gt_bbox_dict
                    seq.frames_dir = os.path.join(self.frames_dir, clip_uid)

                    sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.sequence_list)


class EGO4DLTTrackingDataset(torch.utils.data.Dataset):
    """EGO4D LTT dataset."""

    def __init__(self, data_dir, annotation_path, ratio=1.0, split=None):
        """
        ratio: what percent of the dataset to load, used mostly for testing purpose
        pre_download: pre-download frames from manifold, this is intended to work in FB only
        """
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        dataset = _EGO4DLTTrackingDataset(data_dir, annotation_path, split)
        # Only load a portion of the dataset
        n_total = int(len(dataset) * ratio) + 1
        self.sequences = dataset.sequences[:n_total]
        self.sequence_list = dataset.sequence_list[:n_total]

    def __getitem__(self, index):
        seq = self.sequences[index]

        return seq

    def __len__(self):
        return len(self.sequence_list)

import sys
import os

import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tracking.metrics.lt_tracking_metrics import compute_f_score, compute_precision_and_recall
from tracking.metrics.miou import compute_overlaps
from tracking.dataset.eval_datasets.base_dataset import Sequence, BaseDataset, SequenceList
import argparse
import csv
from tqdm import tqdm
import json

class _EGO4DLTTrackingDataset(BaseDataset):
    """EGO4D Long-term Tracking dataset."""

    def __init__(self, annotation_path, data_dir="/checkpoint/haotang/data/EgoTracks/clips_frames/", split=None):
        super().__init__()
        self.data_dir = data_dir
        self.frames_dir = os.path.join(data_dir, "frames")
        self.info_dir = os.path.join(data_dir, "clip_info")
        self.annotation_path = annotation_path

        self.clip_info = self.load_clip_info(os.path.join(data_dir, "clip_info.json"))
        self.sequences = self.get_sequences()
        self.sequence_list = [seq.name for seq in self.sequences]
        self.split = split

    def get_sequences(self):
        with open(self.annotation_path, "r") as f:
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
        with open(clip_info_path) as f:
            clip_info = json.load(f)

        return clip_info

    def get_lt_track(
        self, video_uid: str, clip_uid: str, frame_number_key: str = "exported_clip_frame_number"
    ):
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
                    if not query_set["is_valid"]:
                        continue
                    if "lt_track" not in query_set:
                        continue

                    # Cannot find the extracted frame dir
                    frames = self.clip_info[clip_uid]["frames"]
                    frames = [
                        os.path.join(self.frames_dir, clip_uid, f) for f in frames
                    ]
                    if frames is None:
                        continue

                    object_title = query_set["object_title"]
                    # visual_crop
                    visual_crop = query_set["visual_crop"]
                    visual_crop_fn = visual_crop[frame_number_key]
                    visual_crop = {
                        visual_crop[frame_number_key]: [
                            visual_crop["x"],
                            visual_crop["y"],
                            visual_crop["width"],
                            visual_crop["height"],
                        ]
                    }

                    # since some annotation does not start with frame #0, so we have to guess from visual crop
                    frame_number_5FPS = list(range(visual_crop_fn, -1, -6))[::-1] + list(range(visual_crop_fn, len(frames), 6))[1:]

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
                    seq.frame_number_5FPS = frame_number_5FPS
                    seq.mapping_30FPS_to_5FPS = {
                        frame_number_5FPS[i]: i for i in range(len(frame_number_5FPS))
                    }
                    seq.mapping_5FPS_to_30FPS = {
                        i: frame_number_5FPS[i] for i in range(len(frame_number_5FPS))
                    }
                    seq.video_uid = video_uid

                    sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.sequence_list)


parser = argparse.ArgumentParser(description='Eval a submission to EgoTracks.')
parser.add_argument('submission_path', type=str,
                    help='Path to the submission directory')
parser.add_argument('--annotation_path', type=str, default="/checkpoint/weiyaowang/tracking/data/ego4d/lt_tracking_val_new.json",
                    help='Path to the submission directory')


def check_missing_sequence(submission_seq_names: set, seq_names: set):
    for s in seq_names:
        if s not in submission_seq_names:
            raise f"Missing sequence {s}"


def load_annotation(annotation_path: str):
    return _EGO4DLTTrackingDataset(annotation_path)


# def load_submission(submission_path:str):
#     res = {}
#     files = os.listdir(submission_path)
#     for f in tqdm(files, total=len(files)):
#         r = []
#         seq_name = f.split(".csv")[0]
#         with open(os.path.join(submission_path, f), "r") as f:
#             csv_reader = csv.DictReader(f)
#             for row in csv_reader:
#                 r.append([int(row["frame_number"]), float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), float(row["confidence"])])
#         res[seq_name] = r

#     return res

def load_submission(submission_path:str):
    with open(submission_path, "r") as f:
        submission = json.load(f)

    return submission


def check_missing_frames(submission: dict, annotation):
    for seq in annotation.sequences:
        name = seq.name
        if len(submission[name]) != len(seq.frames):
            raise f"Missing frames from {seq}"


def compute_competition_metrics(submission, annotation):
    iou_per_video = []
    all_pred_bboxes = []
    all_gt_bboxes = []
    all_pred_scores = []
    all_ious = []
    problematic_seq = set()
    for seq in tqdm(annotation.sequences, total=len(annotation.sequences)):
        target_frame_number = list(seq.visual_crop.keys())[0]

        exclude_frame_numbers = [target_frame_number]
        exclude_frame_numbers = set(exclude_frame_numbers)

        pred_bbox_dict = {int(frame_number): {
            "bbox": bbox[:-1], "score": bbox[-1]
        } for frame_number, bbox in submission[seq.name].items()}
        total_frame_numbers = len(seq.frame_number_5FPS)
        mapping_30FPS_to_5FPS = seq.mapping_30FPS_to_5FPS
        mapping_5FPS_to_30FPS = seq.mapping_5FPS_to_30FPS
        ious = []
        gt_bboxes = [None for _ in range(total_frame_numbers)]
        pred_bboxes = [None for _ in range(total_frame_numbers)]
        pred_scores = [0 for _ in range(total_frame_numbers)]
        for frame_number, gt_bbox in seq.gt_bbox_dict.items():
            frame_number = int(frame_number)
            # Ignore excluded frames.
            if frame_number in exclude_frame_numbers:
                continue

            if frame_number not in mapping_30FPS_to_5FPS:
                problematic_seq.add(seq.video_uid)
                # print(f"{seq.name} {frame_number} not in required frame numbers")
                continue
            gt_bboxes[mapping_30FPS_to_5FPS[frame_number]] = gt_bbox
            if frame_number not in pred_bbox_dict:
                ious.append(0)
            else:
                pred_bbox = pred_bbox_dict[frame_number]["bbox"]
                pred_score = pred_bbox_dict[frame_number]["score"]
                iou = compute_overlaps([gt_bbox], [pred_bbox])[0]
                ious.append(iou)
                pred_bboxes[mapping_30FPS_to_5FPS[frame_number]] = pred_bbox
                pred_scores[mapping_30FPS_to_5FPS[frame_number]] = pred_score

        for fn_5FPS, fn_30FPS in mapping_5FPS_to_30FPS.items():
            if fn_30FPS not in pred_bbox_dict:
                raise f"No prediction for frame {fn_30FPS} in {seq}."

        for fn_5FPS, fn_30FPS in mapping_5FPS_to_30FPS.items():
            pred_bboxes[fn_5FPS] = pred_bbox_dict[fn_30FPS]["bbox"]
            pred_scores[fn_5FPS] = pred_bbox_dict[fn_30FPS]["score"]
            
        if len(ious):
            iou_per_video.append(np.mean(ious))
            all_ious.extend(ious)

        all_gt_bboxes.append(gt_bboxes)
        all_pred_bboxes.append(pred_bboxes)
        all_pred_scores.append(pred_scores)

    average_overlap = np.mean(iou_per_video)
    precision, recall = compute_precision_and_recall(
        all_pred_scores, all_pred_bboxes, all_gt_bboxes
    )
    f_score, pr_score, re_score = compute_f_score(precision, recall)

    return {"f_score": f_score, "pr": pr_score, "re": re_score, "AO": average_overlap}


def main():
    args = parser.parse_args()
    annotation = load_annotation(args.annotation_path)
    submission = load_submission(args.submission_path)
    seq_names = set([anno.name for anno in annotation.sequences])
    check_missing_sequence(submission.keys(), seq_names)
    # check_missing_frames(submission, annotation)
    res = compute_competition_metrics(submission, annotation)
    print(res)

    return res["f_score"]

if __name__ == "__main__":
    main()
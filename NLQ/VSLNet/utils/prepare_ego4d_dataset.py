#! /usr/bin/env python
"""
Prepare Ego4d episodic memory NLQ for model training.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import csv
import json
import math
import os

import torch
import tqdm


CANONICAL_VIDEO_FPS = 30.0
FEATURE_WINDOW_SIZE = 16.0
FEATURES_PER_SEC = CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE


def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE))


def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"


def reformat_data(split_data, test_split=False):
    """Convert the format from JSON files.
    fps, num_frames, timestamps, sentences, exact_times,
    annotation_uids, query_idx.
    """
    formatted_data = {}
    clip_video_map = {}
    for video_datum in split_data["videos"]:
        for clip_datum in video_datum["clips"]:
            clip_uid = clip_datum["clip_uid"]
            clip_video_map[clip_uid] = (
                video_datum["video_uid"],
                clip_datum["video_start_sec"],
                clip_datum["video_end_sec"],
            )
            clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
            num_frames = get_nearest_frame(clip_duration, math.ceil)
            new_dict = {
                "fps": FEATURES_PER_SEC,
                "num_frames": num_frames,
                "timestamps": [],
                "exact_times": [],
                "sentences": [],
                "annotation_uids": [],
                "query_idx": [],
            }

            for ann_datum in clip_datum["annotations"]:
                for index, datum in enumerate(ann_datum["language_queries"]):
                    if not test_split:
                        start_time = float(datum["clip_start_sec"])
                        end_time = float(datum["clip_end_sec"])
                    else:
                        # Random placeholders for test set.
                        start_time = 0.
                        end_time = 0.

                    if "query" not in datum or not datum["query"]:
                        continue
                    new_dict["sentences"].append(process_question(datum["query"]))
                    new_dict["annotation_uids"].append(ann_datum["annotation_uid"])
                    new_dict["query_idx"].append(index)
                    new_dict["exact_times"].append([start_time, end_time]),
                    new_dict["timestamps"].append(
                        [
                            get_nearest_frame(start_time, math.floor),
                            get_nearest_frame(end_time, math.ceil),
                        ]
                    )
            formatted_data[clip_uid] = new_dict
    return formatted_data, clip_video_map


def convert_ego4d_dataset(args):
    """Convert the Ego4D dataset for VSLNet."""
    # Reformat the splits to train vslnet.
    all_clip_video_map = {}
    for split in ("train", "val", "test"):
        read_path = args[f"input_{split}_split"]
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        data_split, clip_video_map = reformat_data(raw_data, split == "test")
        all_clip_video_map.update(clip_video_map)
        num_instances = sum(len(ii["sentences"]) for ii in data_split.values())
        print(f"# {split}: {num_instances}")

        os.makedirs(args["output_save_path"], exist_ok=True)
        save_path = os.path.join(args["output_save_path"], f"{split}.json")
        print(f"Writing [{split}]: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(data_split, file_id)

    # Extract visual features based on the all_clip_video_map.
    feature_sizes = {}
    os.makedirs(args["clip_feature_save_path"], exist_ok=True)
    progress_bar = tqdm.tqdm(all_clip_video_map.items(), desc="Extracting features")
    for clip_uid, (video_uid, start_sec, end_sec) in progress_bar:
        feature_path = os.path.join(args["video_feature_read_path"], f"{video_uid}.pt")
        feature = torch.load(feature_path)

        # Get the lower frame (start_sec) and upper frame (end_sec) for the clip.
        clip_start = get_nearest_frame(start_sec, math.floor)
        clip_end = get_nearest_frame(end_sec, math.ceil)
        clip_feature = feature[clip_start : clip_end + 1]
        feature_sizes[clip_uid] = clip_feature.shape[0]
        feature_save_path = os.path.join(
            args["clip_feature_save_path"], f"{clip_uid}.pt"
        )
        torch.save(clip_feature, feature_save_path)

    save_path = os.path.join(args["clip_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_train_split", required=True, help="Path to Ego4d train split"
    )
    parser.add_argument(
        "--input_val_split", required=True, help="Path to Ego4d val split"
    )
    parser.add_argument(
        "--input_test_split", required=True, help="Path to Ego4d test split"
    )
    parser.add_argument(
        "--output_save_path", required=True, help="Path to save the output jsons"
    )
    parser.add_argument(
        "--video_feature_read_path", required=True, help="Path to read video features"
    )
    parser.add_argument(
        "--clip_feature_save_path",
        required=True,
        help="Path to save clip video features",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    convert_ego4d_dataset(parsed_args)

import argparse
import gzip
import json
import os
import random

import tqdm


def get_dataset_uid(split, idx):
    return f"{split}_{idx:010d}"


def process_video_annot(video_data, n_samples_so_far):
    annotations = []
    split = video_data["split"]
    for clip_data in video_data["clips"]:
        for clip_annot in clip_data["annotations"]:
            for qset_id, qset in clip_annot["query_sets"].items():
                if not qset["is_valid"]:
                    continue
                curr_annot = {
                    "metadata": {
                        "video_uid": video_data["video_uid"],
                        "video_start_sec": clip_data["video_start_sec"],
                        "video_end_sec": clip_data["video_end_sec"],
                        "clip_fps": clip_data["clip_fps"],
                    },
                    "clip_uid": clip_data["clip_uid"],
                    "query_set": qset_id,
                    "query_frame": qset["query_frame"],
                    "response_track": qset["response_track"],
                    "visual_crop": qset["visual_crop"],
                    "object_title": qset["object_title"],
                    # Assign a unique ID to this annotation for the dataset
                    "dataset_uid": get_dataset_uid(split, n_samples_so_far),
                }
                annotations.append(curr_annot)
                n_samples_so_far = n_samples_so_far + 1
    return annotations


def process_data(args):
    annotation_exports = {
        "train": json.load(
            open(
                os.path.join(args.annot_root, "vq_train.json"),
                "r",
            )
        ),
        "val": json.load(
            open(
                os.path.join(args.annot_root, "vq_val.json"),
                "r",
            )
        ),
    }

    os.makedirs(args.save_root, exist_ok=True)
    for split, split_export in annotation_exports.items():
        n_samples = 0
        split_data = []
        print(f"========> Processing {split} data")
        for video_data in tqdm.tqdm(split_export["videos"]):
            proc_video_data = process_video_annot(video_data, n_samples)
            split_data += proc_video_data
            n_samples += len(proc_video_data)
        save_path = os.path.join(args.save_root, f"{split}_annot.json.gz")
        with gzip.open(save_path, "wt") as fp:
            json.dump(split_data, fp)
        # Print dataset statistics
        uniq_videos = set()
        uniq_clips = set()
        n_annots = 0
        for data in split_data:
            uniq_videos.add(data["metadata"]["video_uid"])
            uniq_clips.add(data["clip_uid"])
            n_annots += 1
        print(f"# videos     : {len(uniq_videos)}")
        print(f"# clips      : {len(uniq_clips)}")
        print(f"# annotations: {n_annots}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-root", type=str, default="data")
    parser.add_argument("--save-root", type=str, default="./")

    args = parser.parse_args()

    random.seed(123)
    process_data(args)

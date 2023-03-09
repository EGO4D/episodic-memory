"""
Extract frames from clip
"""

import csv
import functools
import json
import multiprocessing
import os
import time
from collections import defaultdict
from typing import List, NamedTuple

import av

from PIL import Image
from tqdm import tqdm


class ExtractFramesWorkflowParams(NamedTuple):
    num_process: int = 16
    clip_dir: str = "/datasets01/ego4d_track2/v1/clips"
    annotation_path: str = ("/checkpoint/haotang/data/EgoTracks/annotations/train_v1.json")
    output_dir: str = "/checkpoint/haotang/data/EgoTracks/clips_frames"
    annotated_frames_only: bool = True


def run_single_process(data: list, params: ExtractFramesWorkflowParams):
    clip_uid, annotated_frame_numbers = data
    frames_save_dir = os.path.join(params.output_dir, "frames", f"{clip_uid}")
    info_save_dir = os.path.join(params.output_dir, "clip_info")
    info_save_path = os.path.join(
        params.output_dir, "clip_info", f"{clip_uid}.csv"
    )
    os.makedirs(frames_save_dir, exist_ok=True)
    os.makedirs(info_save_dir, exist_ok=True)
    clip_path = os.path.join(params.clip_dir, f"{clip_uid}.mp4")
    frame_numbers = []

    print(f"Start processing {clip_uid}!")
    s = time.time()
    with av.open(clip_path) as container:
        avg_fps = container.streams.video[0].average_rate
        stream_base = container.streams.video[0].time_base
        pts_scale = avg_fps * stream_base

        for frame in container.decode(video=0):
            frame_number = int(frame.pts * pts_scale)
            if not params.annotated_frames_only or frame_number in annotated_frame_numbers:
                image = frame.to_ndarray(format="rgb24")
                pil_img = Image.fromarray(image)
                pil_img.save(
                    os.path.join(frames_save_dir, f"{frame_number}.jpg"), format="JPEG"
                )
                # cv2.imwrite(os.path.join(local_output_dir, f"{frame_number}.jpg"), image)
                frame_numbers.append(f"{frame_number}.jpg")


    with open(info_save_path, "w") as f:
        write = csv.writer(f, delimiter="\n")
        write.writerow(frame_numbers)

    print(f"Finished {clip_uid} in {time.time() - s}!")


def extract_clip_ids(file_path: str):
    with open(file_path, "r") as f:
        annotations = json.load(f)
    clip_uids = []
    annotated_frame_numbers = []
    ignored = 0
    for v in tqdm(annotations["videos"], total=len(annotations["videos"])):
        for c in v["clips"]:
            fns = set()
            has_problem = False
            assert len(c["annotations"]) == 1
            for k, query_set in c["annotations"][0]["query_sets"].items():
                # some VQ are not in, and some has problem with frame number conversion, ignore
                if "lt_track" not in query_set or "exported_clip_frame_number" not in query_set["visual_crop"]:
                    has_problem = True
                    continue
                fns.add(query_set["visual_crop"]["exported_clip_frame_number"])
                for track in ["response_track", "lt_track"]:
                    for bbox_dict in query_set[track]:
                        if "exported_clip_frame_number" not in bbox_dict:
                            has_problem = True
                            break
                        fns.add(bbox_dict["exported_clip_frame_number"])
            
            if not has_problem:
                clip_uids.append(c["exported_clip_uid"])
                annotated_frame_numbers.append(fns)
            else:
                ignored += 1

    return clip_uids, annotated_frame_numbers


def remove_finished_clip_uids(clip_uids: List, params: ExtractFramesWorkflowParams):
    res = []
    info_save_dir = os.path.join(params.output_dir, "clip_info")

    for clip_uid in clip_uids:
        if not os.path.exists(os.path.join(info_save_dir, f"{clip_uid}.csv")):
            res.append(clip_uid)
        else:
            print(f"{clip_uid} was already extracted!")

    return res


def read_csv(path: str):
    if not os.path.exists(path):
        raise RuntimeError
    with open(path) as f:
        frame_numbers = [line.strip() for line in f.readlines()]
    return frame_numbers


def combine_clip_info(params):
    combined_save_path = os.path.join(params.output_dir, "clip_info.json")
    clip_info_dir = os.path.join(params.output_dir, "clip_info")
    clip_info_files = os.listdir(clip_info_dir)
    clip_info_dict = defaultdict(dict)
    for clip_info_file in tqdm(clip_info_files, total=len(clip_info_files)):
        clip_uid = clip_info_file.split(".csv")[0]

        info_path = os.path.join(clip_info_dir, f"{clip_uid}.csv")
        frame_numbers = read_csv(info_path)
        clip_info_dict[clip_uid]["frames"] = frame_numbers

    with open(combined_save_path, "w") as f:
        json.dump(clip_info_dict, f)


def main():
    params = ExtractFramesWorkflowParams()
    clip_uids, annotated_frame_numbers = extract_clip_ids(params.annotation_path)
    clip_uids = remove_finished_clip_uids(clip_uids, params)
    print(f"Total {len(clip_uids)} to be processed ...")

    pool = multiprocessing.Pool(params.num_process)
    pool.map(functools.partial(run_single_process, params=params), zip(clip_uids, annotated_frame_numbers))

    pool.close()
    pool.join()
    # Combine info for each clip into one file
    combine_clip_info(params)


if __name__ == "__main__":
    main()

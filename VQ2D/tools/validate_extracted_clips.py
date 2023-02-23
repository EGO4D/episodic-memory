"""
Script to extract clips from a video
"""
import argparse
import json
import os

import pims
import tqdm
from vq2d.baselines import get_clip_name_from_clip_uid


def approx_equal_durations(dur1, dur2, thresh=1.0):
    return abs(dur1 - dur2) < thresh


def check_clip(clip_uid, clip_annotations, clip_metadata, args):
    video_start_sec = clip_metadata["video_start_sec"]
    video_end_sec = clip_metadata["video_end_sec"]
    # Check if extracted clip duration matches expected duration
    expected_duration = video_end_sec - video_start_sec
    clip_path = os.path.join(args.clips_root, get_clip_name_from_clip_uid(clip_uid))
    with pims.Video(clip_path) as clip_reader:
        n_clip_frames = len(clip_reader)
        actual_duration = float(n_clip_frames) / clip_reader.frame_rate
    assert approx_equal_durations(
        expected_duration, actual_duration
    ), f"Expected duration {expected_duration} for {clip_path} but got {actual_duration}"
    # Check if annotations are within range of clip frames
    for a in clip_annotations:
        for qset_id, qset in a["query_sets"].items():
            if not qset["is_valid"]:
                continue
            vc_fno = qset["visual_crop"]["frame_number"]
            rt_fnos = [rf["frame_number"] for rf in qset["response_track"]]
            q_fno = qset["query_frame"]
            assert (
                vc_fno < n_clip_frames
            ), "VC fno {} is out of range of clip {} with {} frames".format(
                vc_fno, clip_uid, n_clip_frames
            )
            assert (
                q_fno < n_clip_frames
            ), "Query fno {} is out of range of clip {} with {} frames".format(
                q_fno, clip_uid, n_clip_frames
            )
            for rt_fno in rt_fnos:
                assert (
                    rt_fno < n_clip_frames
                ), "Response fno {} is out of range of clip {} with {} frames".format(
                    rt_fno, clip_uid, n_clip_frames
                )


def main(args):
    # Load annotations
    annotation_export = []
    for annot_path in args.annot_paths:
        annotation_export += json.load(open(annot_path, "r"))["videos"]
    # Get clip-wise annotations
    clips2annotations = {}
    clip2metadata = {}
    for v in annotation_export:
        v_uid = v["video_uid"]
        for c in v["clips"]:
            c_uid = c["clip_uid"]
            if c_uid is None:
                continue
            clip2metadata[c_uid] = {
                "video_uid": v_uid,
                **{k: v for k, v in c.items() if k not in ["clip_uid", "annotations"]},
            }
            if c_uid not in clips2annotations:
                clips2annotations[c_uid] = []
            clips2annotations[c_uid] += c["annotations"]
    for c_uid in tqdm.tqdm(sorted(clips2annotations.keys())):
        try:
            check_clip(c_uid, clips2annotations[c_uid], clip2metadata[c_uid], args)
        except AssertionError as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-paths", type=str, required=True, nargs="+")
    parser.add_argument("--clips-root", type=str, required=True)
    args = parser.parse_args()

    main(args)

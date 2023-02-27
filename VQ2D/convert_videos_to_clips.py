"""
Script to extract clips from a video
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess as sp

import cv2
import imageio
import pims
import tqdm
from vq2d.baselines import get_clip_name_from_clip_uid


def read_video_md(path):
    with imageio.get_reader(path, format="mp4") as reader:
        metadata = reader.get_meta_data()
    return metadata


def get_mp4_writer(path, fps, output_params=["-crf", "22"]):
    writer = imageio.get_writer(
        path,
        codec="h264",
        fps=fps,
        quality=None,
        pixelformat="yuv420p",
        bitrate=0,  # Setting bitrate to 0 is required to activate -crf
        macro_block_size=None,
        output_params=output_params,
    )
    return writer


def frames_to_select(
    start_frame: int,
    end_frame: int,
    original_fps: int,
    new_fps: int,
):
    # ensure the new fps is divisible by the old
    assert original_fps % new_fps == 0

    # check some obvious things
    assert end_frame >= start_frame

    num_frames = end_frame - start_frame + 1
    skip_number = original_fps // new_fps
    for i in range(0, num_frames, skip_number):
        yield i + start_frame


def approx_equal_durations(dur1, dur2, thresh=1.0):
    return abs(dur1 - dur2) < thresh


def extract_clip(video_path, clip_data, save_root, downscale_height=700):
    """
    Extracts clips from a video
    Save path format: {save_root}/{clip_uid}.mp4

    Args:
        video_path - path to video
        clip_data - a clip annotation from the VQ task export
        save_root - path to save extracted images
    """
    clip_uid = clip_data["clip_uid"]
    if clip_uid is None:
        return None
    clip_save_path = os.path.join(save_root, get_clip_name_from_clip_uid(clip_uid))
    video_md = read_video_md(video_path)
    if os.path.isfile(clip_save_path):
        # If file exists, try loading video metadata
        try:
            # Metadata read success
            with pims.Video(clip_save_path) as test_reader:
                actual_clip_duration = float(len(test_reader)) / test_reader.frame_rate
            expected_clip_duration = (
                clip_data["video_end_sec"] - clip_data["video_start_sec"]
            )
            if not approx_equal_durations(actual_clip_duration, expected_clip_duration):
                print(actual_clip_duration, expected_clip_duration)
            assert approx_equal_durations(actual_clip_duration, expected_clip_duration)
            return None
        except Exception as e:
            # Metadata read failed
            print(f"Clip extraction incomplete for {clip_save_path}. Recreating.")
            sp.call(["rm", clip_save_path])
    # Select frames for clip
    clip_fps = int(clip_data["clip_fps"])
    video_fps = int(video_md["fps"])
    vsf = clip_data["video_start_frame"]
    vef = clip_data["video_end_frame"]
    reader = pims.Video(video_path)
    # Downscale images to save memory
    frame = reader[0]
    if downscale_height > 0:
        frame_scale = float(downscale_height) / frame.shape[0]
        new_H = downscale_height
        new_W = int(frame.shape[1] * frame_scale)
        if new_W % 2 == 1:  # ffmpeg requirement
            new_W += 1
    # Create video clip
    with get_mp4_writer(clip_save_path, clip_fps) as writer:
        for fno in frames_to_select(vsf, vef, video_fps, clip_fps):
            try:
                frame = reader[fno]
            except:
                max_fno = int(video_md["fps"] * video_md["duration"])
                print(
                    f"===> frame {fno} out of range for video {video_path} (max fno = {max_fno})"
                )
                break
            if downscale_height > 0:
                frame = cv2.resize(frame, (new_W, new_H))
            writer.append_data(frame)
    reader.close()


def batchify_video_uids(video_uids, batch_size):
    video_uid_batches = []
    nbatches = len(video_uids) // batch_size
    if batch_size * nbatches < len(video_uids):
        nbatches += 1
    for batch_ix in range(nbatches):
        video_uid_batches.append(
            video_uids[batch_ix * batch_size : (batch_ix + 1) * batch_size]
        )
    return video_uid_batches


def video_to_clip_fn(inputs):
    video_data, args = inputs
    video_uid = video_data["video_uid"]
    video_path = os.path.join(args.ego4d_videos_root, video_uid + ".mp4")
    if not os.path.isfile(video_path):
        print(f"Missing video {video_path}")
        return None

    for clip_data in video_data["clips"]:
        if args.clip_uids is not None and clip_data["clip_uid"] not in args.clip_uids:
            continue
        extract_clip(
            video_path,
            clip_data,
            args.save_root,
            downscale_height=args.downscale_height,
        )


def main(args):
    # Load annotations
    annotation_export = []
    for annot_path in args.annot_paths:
        annotation_export += json.load(open(annot_path, "r"))["videos"]
    video_uids = sorted([a["video_uid"] for a in annotation_export])
    os.makedirs(args.save_root, exist_ok=True)
    if args.video_batch_idx >= 0:
        video_uid_batches = batchify_video_uids(video_uids, args.video_batch_size)
        video_uids = video_uid_batches[args.video_batch_idx]
        print(f"===> Processing video_uids: {video_uids}")
    # Get annotations corresponding to video_uids
    annotation_export = [a for a in annotation_export if a["video_uid"] in video_uids]

    pool = mp.Pool(args.num_workers)
    inputs = [(video_data, args) for video_data in annotation_export]
    _ = list(
        tqdm.tqdm(
            pool.imap_unordered(video_to_clip_fn, inputs),
            total=len(inputs),
            desc="Converting videos to clips",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-batch-idx", type=int, default=-1)
    parser.add_argument("--annot-paths", type=str, required=True, nargs="+")
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--ego4d-videos-root", type=str, required=True)
    parser.add_argument("--video-batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--clip-uids", type=str, nargs="+", default=None)
    parser.add_argument("--downscale-height", type=int, default=-1)
    args = parser.parse_args()

    main(args)

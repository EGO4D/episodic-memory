"""
Script to extract images from a video
"""
import argparse
import collections
import json
import multiprocessing as mp
import os

import imageio
import pims
import tqdm
from vq2d.baselines.utils import get_image_name_from_clip_uid


def read_video_md(path):
    with imageio.get_reader(path, format="mp4") as reader:
        metadata = reader.get_meta_data()
    return metadata


def save_video_frames(path, frames_to_save):
    video_md = read_video_md(path)
    frames_to_save_dict = collections.defaultdict(list)
    for fs in frames_to_save:
        frames_to_save_dict[fs["video_fno"]].append(fs["save_path"])
    reader = pims.Video(path)
    for fno, paths in frames_to_save_dict.items():
        try:
            f = reader[fno]
        except:
            max_fno = int(video_md["fps"] * video_md["duration"])
            print(
                f"===> frame {fno} out of range for video {path} (max fno = {max_fno})"
            )
            continue
        for path in paths:
            if not os.path.isfile(path):
                imageio.imwrite(path, f)


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


def extract_clip_frame_nos(video_md, clip_annotation, save_root):
    """
    Extracts frame numbers corresponding to the VQ annotation for a given clip

    Args:
        video_md - a dictionary of video metadata
        clip_annotation - a clip annotation from the VQ task export
        save_root - path to save extracted images
    """
    clip_uid = clip_annotation["clip_uid"]
    clip_fps = int(clip_annotation["clip_fps"])
    # Select frames for clip
    video_fps = int(video_md["fps"])
    vsf = clip_annotation["video_start_frame"]
    vef = clip_annotation["video_end_frame"]
    video_frames_for_clip = list(frames_to_select(vsf, vef, video_fps, clip_fps))
    # Only save images containing response_track and visual_crop
    annotation = clip_annotation["annotations"][0]
    frames_to_save = []
    for qset_id, qset in annotation["query_sets"].items():
        if not qset["is_valid"]:
            continue
        vc_fno = qset["visual_crop"]["frame_number"]
        rt_fnos = [rf["frame_number"] for rf in qset["response_track"]]
        all_fnos = [vc_fno] + rt_fnos
        for fno in all_fnos:
            path = os.path.join(save_root, get_image_name_from_clip_uid(clip_uid, fno))
            if os.path.isfile(path):
                continue
            frames_to_save.append(
                {"video_fno": video_frames_for_clip[fno], "save_path": path}
            )
    return frames_to_save


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


def video_to_image_fn(inputs):
    video_data, args = inputs
    video_uid = video_data["video_uid"]

    # Extract frames for a specific video_uid
    video_path = os.path.join(args.ego4d_videos_root, video_uid + ".mp4")
    if not os.path.isfile(video_path):
        print(f"Missing video {video_path}")
        return None

    # Get list of frames to save for annotated clips
    video_md = read_video_md(video_path)
    frame_nos_to_save = []
    for clip_data in video_data["clips"]:
        if clip_data["clip_uid"] is None:
            continue
        # Create root directory to save clip
        os.makedirs(os.path.join(args.save_root, clip_data["clip_uid"]), exist_ok=True)
        # Get list of frames to save
        frame_nos_to_save += extract_clip_frame_nos(video_md, clip_data, args.save_root)

    if len(frame_nos_to_save) == 0:
        print(f"=========> No valid frames to read for {video_uid}!")
        return None

    save_video_frames(video_path, frame_nos_to_save)


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
            pool.imap_unordered(video_to_image_fn, inputs),
            total=len(inputs),
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
    args = parser.parse_args()

    main(args)

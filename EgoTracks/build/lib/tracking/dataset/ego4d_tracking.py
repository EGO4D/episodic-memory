import json
import logging
import os
from itertools import groupby
from typing import Dict

import av
import numpy as np
import torch
from tracking.utils.env import pathmgr
from tracking.utils.utils import extract_frames, extract_frames_by_range


class Ego4DTracking(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_path: str,
        clip_dir: str,
        video_dir: str,
        is_read_5FPS_clip: bool = True,
        return_5FPS_frames: bool = True,
    ):
        """
        Args:
            clip_dir: Path to the directory that stores the clips. Those are the raw 5FPS clips for the VQ annotation.
            video_dir: Path to the directory that stores the videos. Those are the raw 30FPS videos in Ego4D.
            is_read_5FPS_clip: Whether to read from VQ clip (5FPS) or video (30FPS)
            return_5FPS_frames: Only used if is_read_5FPS_clip is False. If return_5FPS_frames is True, we only return frames at 5FPS,
                but those frames are extracted from the 30FPS video.
        """
        # TODO: Very strange, when using dataloader, the pathmgr will
        # loss the ManifoldHandler. Temporarily remember pathmgr so it
        # can be used in download_clip_locally.
        self.pathmgr = pathmgr
        with pathmgr.open(annotation_path, "r") as f:
            vq_annotations = json.load(f)
            vq_ann_video_uids = [x["video_uid"] for x in vq_annotations["videos"]]

        self.vq_video_dict = {
            x["video_uid"]: x["clips"] for x in vq_annotations["videos"]
        }
        self.annotation_path = annotation_path
        self.clip_dir = clip_dir
        self.video_dir = video_dir
        self.is_read_5FPS_clip = is_read_5FPS_clip
        self.return_5FPS_frames = return_5FPS_frames
        self.response_tracks = []

        # Load annotation all the visual query, including response track and visual crop.
        for video_uid in vq_ann_video_uids:
            clips = self.vq_video_dict[video_uid]
            for clip in clips:
                clip_uid = clip["source_clip_uid"]

                if self.is_read_5FPS_clip:
                    # Since we will load by at the clip level, not the video level, we load by "frame_number"
                    response_tracks = self.get_VQ_response_track_by_video_clip_uid(
                        video_uid, clip_uid, frame_number_key="frame_number"
                    )
                else:
                    # If we want to use original 30FPS video, we need to read the video_frame_number
                    response_tracks = self.get_VQ_response_track_by_video_clip_uid(
                        video_uid, clip_uid, frame_number_key="video_frame_number"
                    )
                self.response_tracks.extend(response_tracks)
        self.response_tracks = self.response_tracks

    def __len__(self):
        return len(self.response_tracks)

    def __getitem__(self, index: int):
        response_track = self.response_tracks[index]
        video_uid, clip_uid, response_track, object_title, target_id, visual_crop = (
            response_track["video_uid"],
            response_track["clip_uid"],
            response_track["response_track"],
            response_track["object_title"],
            response_track["target_id"],
            response_track["visual_crop"],
        )

        visual_crop_frame_number = visual_crop["frame"]
        visual_crop_bbox = [
            visual_crop["x"],
            visual_crop["y"],
            visual_crop["width"],
            visual_crop["height"],
        ]
        imgs = []
        frame_bbox_dict = {}
        frame_numbers = []
        for frame, data in groupby(response_track, lambda x: x["frame"]):
            data = list(data)
            assert len(data) == 1

            for b in data:
                # img = frames[0].to_ndarray(format="bgr24")
                box = [b["x"], b["y"], b["width"], b["height"]]
                frame_bbox_dict[frame] = box
                frame_numbers.append(frame)

        if self.is_read_5FPS_clip:
            clip_path = self.download_clip_locally(clip_uid)
            with av.open(clip_path) as container:
                imgs = extract_frames(frame_numbers, container)

            with av.open(clip_path) as container:
                visual_crop_image = extract_frames(
                    [visual_crop_frame_number], container
                )

                # This happens rarely, some clip have missing frames, but duplicated in the later frame_number
                # Search for the next frame instead
                # TODO: This is hacky ...
                if len(visual_crop_image) == 0:
                    with av.open(clip_path) as container:
                        visual_crop_image = extract_frames(
                            [visual_crop_frame_number + 1], container
                        )
        else:
            video_path = self.download_video_locally(video_uid)
            video_frame_numbers = list(
                range(min(frame_numbers), max(frame_numbers) + 1)
            )
            with av.open(video_path) as container:
                imgs = extract_frames_by_range(
                    video_frame_numbers[0], video_frame_numbers[-1], container
                )
            with av.open(video_path) as container:
                visual_crop_image = extract_frames(
                    [visual_crop_frame_number], container
                )

            # override previous list by 30FPS frame numbers
            frame_numbers = video_frame_numbers

        # This should not happen, but some clip has duplicated frame_number
        # E.g. clip_uid: c5eab264-e780-4aa1-a89b-07b9c090cdc2, the last several frames are:
        # ... 950, 951, 952, 954, 954. It misses 953 and 954 is duplicated
        # Just return the last one extracted.
        if len(visual_crop_image) != 1:
            logging.error(
                f"Returned # of visual_crop_image is not correct. {clip_uid}, # of visual_crop {len(visual_crop_image)}, frame_number {visual_crop_frame_number}, index {index}"
            )

        visual_crop_image = visual_crop_image[-1]

        # This is only in effect when is_read_5FPS_clip = False
        if self.return_5FPS_frames:
            new_imgs = []
            new_frame_numbers = []
            for i, frame_number in enumerate(frame_numbers):
                # this should not happen. logging any error
                if i >= len(imgs):
                    logging.error(
                        f"video_uid {video_uid}, clip_uid {clip_uid}, # of frames {len(frame_numbers)} do not match extracted frames {len(imgs)}"
                    )

                if frame_number in frame_bbox_dict and i < len(imgs):
                    new_imgs.append(imgs[i])
                    new_frame_numbers.append(frame_number)

            frame_numbers = new_frame_numbers
            imgs = new_imgs

        imgs = np.array(imgs)
        # reshape to channel, H, W
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        visual_crop_image = (
            torch.from_numpy(visual_crop_image)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return {
            "imgs": imgs,
            "frame_bbox_dict": frame_bbox_dict,
            "frame_numbers": frame_numbers,
            "video_uid": video_uid,
            "clip_uid": clip_uid,
            "object_title": object_title,
            "target_id": target_id,
            "visual_crop": {
                "image": visual_crop_image,
                "bbox": visual_crop_bbox,
                "frame_number": visual_crop_frame_number,
            },
        }

    def download_clip_locally(self, clip_uid: str):
        clip_path = os.path.join(self.clip_dir, f"{clip_uid}.mp4")
        local_path = self.pathmgr.get_local_path(clip_path)
        logging.info(f"Download from {clip_path} to {local_path}.")

        return local_path

    def download_video_locally(self, video_uid: str):
        video_path = os.path.join(self.video_dir, f"{video_uid}")
        local_path = self.pathmgr.get_local_path(video_path)
        logging.info(f"Download from {video_path} to {local_path}.")

        return local_path

    def get_VQ_response_track_by_video_clip_uid(
        self, video_uid: str, clip_uid: str, frame_number_key: str = "frame_number"
    ) -> Dict:
        vq_video_dict = self.vq_video_dict
        ann = vq_video_dict.get(video_uid)
        response_tracks = []

        for clip in ann[:3]:
            if not clip["source_clip_uid"] == clip_uid:
                continue
            for cann in clip["annotations"][:1]:
                for target_id, query_set in cann["query_sets"].items():
                    if not query_set["is_valid"]:
                        continue

                    object_title = query_set["object_title"]
                    visual_crop = query_set["visual_crop"]
                    vq_visual_crop = {
                        "frame": visual_crop[frame_number_key],
                        "x": visual_crop["x"],
                        "y": visual_crop["y"],
                        "width": visual_crop["width"],
                        "height": visual_crop["height"],
                    }
                    response_track = []
                    for frame in query_set["response_track"]:
                        response_track.append(
                            {
                                "frame": frame[frame_number_key],
                                "x": frame["x"],
                                "y": frame["y"],
                                "width": frame["width"],
                                "height": frame["height"],
                            }
                        )
                    response_tracks.append(
                        {
                            "response_track": response_track,
                            "visual_crop": vq_visual_crop,
                            "video_uid": video_uid,
                            "clip_uid": clip_uid,
                            "object_title": object_title,
                            "target_id": target_id,
                        }
                    )

        return response_tracks

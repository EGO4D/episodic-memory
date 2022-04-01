import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

sys.path.append('Camera_Intrinsics_API/')
from get_camera_intrinsics import CameraIntrinsicsHelper


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_frames_for_sfm/',
        help="Input folder with the frames selected for COLMAP.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/sfm/',
        help="Folder for the COLMAP outputs.",
    )
    args = parser.parse_args()

    sfm_images_root = args.input_dir
    sfm_workspace_dir = args.output_dir
    Path(sfm_workspace_dir).mkdir(parents=True, exist_ok=True)

    def run_automatic_reconstruction(inputs):
        video_uid=inputs['video_uid']
        input_images_dir=inputs['input_images_dir']
        sfm_workspace_dir=inputs['sfm_workspace_dir']
        helper = CameraIntrinsicsHelper()
        helper.sfm_workspace_dir=sfm_workspace_dir
        helper.sfm_images_dir = input_images_dir
        l = helper.run_colmap()
        return {
            'result':l,
            'video_uid':video_uid,
        }

    video_uids = os.listdir(sfm_images_root)
    outputs = {}
    for video_uid in video_uids:
        input_images_dir = os.path.join(sfm_images_root,
                                        video_uid)
        output_sfm_dir = os.path.join(sfm_workspace_dir,
                                      video_uid)

        if os.path.isdir(output_sfm_dir):
            continue

        Path(output_sfm_dir).mkdir(parents=True, exist_ok=True)

        r = run_automatic_reconstruction(
            {
                'video_uid':video_uid,
                'input_images_dir':input_images_dir,
                'sfm_workspace_dir':output_sfm_dir,
            }
        )


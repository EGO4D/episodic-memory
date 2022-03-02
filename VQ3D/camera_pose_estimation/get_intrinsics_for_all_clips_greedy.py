import os
import sys
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from multiprocessing import Pool

sys.path.append('Camera_Intrinsics_API/')
from get_camera_intrinsics import CameraIntrinsicsHelper

print('Update colmap path in get_camera_intrinsics.py')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/videos_frames/',
        help="Input folder with the frames selected for COLMAP.",
    )
    parser.add_argument(
        "--sfm_input_dir",
        type=str,
        default='data/videos_frames_for_sfm_greedy/',
        help="Input folder with the frames selected for COLMAP.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/videos_sfm_greedy/',
        help="Folder for the COLMAP outputs.",
    )
    args = parser.parse_args()
    
    images_root = args.input_dir
    
    sfm_images_root = args.sfm_input_dir
    Path(sfm_images_root).mkdir(parents=True, exist_ok=True)
    
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

    video_uids = os.listdir(images_root)
    shuffle(video_uids)
    for video_uid in video_uids:
        # check if processed or in process
        input_images_dir = os.path.join(sfm_images_root,
                                        video_uid)
        output_sfm_dir = os.path.join(sfm_workspace_dir,
                                      video_uid)
        if os.path.isdir(output_sfm_dir):
            continue
        
        Path(output_sfm_dir).mkdir(parents=True, exist_ok=True)
        Path(input_images_dir).mkdir(parents=True, exist_ok=True)


        # copy images to folder
        video_frames_root = os.path.join(images_root,video_uid)
        video_frames_filenames = os.listdir(video_frames_root)
        video_frames_filenames = sorted(video_frames_filenames)
        N = len(video_frames_filenames)

        mid = int(N/2.0)
        for i in range(mid, mid+100,1):
            shutil.copyfile(
                    os.path.join(video_frames_root,video_frames_filenames[i]),
                    os.path.join(input_images_dir,video_frames_filenames[i])
                )

        # run colmap
        r = run_automatic_reconstruction(
            {
                'video_uid':video_uid,
                'input_images_dir':input_images_dir,
                'sfm_workspace_dir':output_sfm_dir,
            }
        )



import os
import argparse
import subprocess as sp
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    type=str,
    default='data/clips_from_videos_camera_poses/',
    help="Camera pose folder"
)

args = parser.parse_args()

pose_dir=args.input_dir

l = os.listdir(pose_dir)

cpt_valid_queries = 0
for clip_uid in tqdm(l):
    # get depth
    depth_dir = os.path.join('../',
                             pose_dir,
                             clip_uid,
                             'egovideo',
                             'depth_DPT_predRT'
                            )

    tmp_input_depth_dir = os.path.join('../',
                                       pose_dir,
                                       clip_uid,
                                       'egovideo',
                                       'input_depth_DPT_predRT',
                                      )

    if os.path.isdir(tmp_input_depth_dir):
        o = sp.check_output(["./main.sh",
                             f"{tmp_input_depth_dir}",
                             f"{depth_dir}",
                            ])


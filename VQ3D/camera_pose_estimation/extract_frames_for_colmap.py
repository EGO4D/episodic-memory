import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

sys.path.append('./Camera_Intrinsics_API/')
from get_camera_intrinsics import CameraIntrinsicsHelper


def extract_frames_for_sfm(inputs):
    images_dir = inputs['images_dir']
    output_dir = inputs['output_dir']
    helper = CameraIntrinsicsHelper()
    helper.sfm_images_dir = output_dir
    helper.select_good_frames(images_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_frames/',
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/clips_frames_for_sfm/',
        help="Output folder with the clips.",
    )
    parser.add_argument(
        "--j",
        type=int,
        default=1,
        help="Number of parallel processes",
    )
    args = parser.parse_args()

    clip_images_dir = args.input_dir
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    inputs=[]
    clips_dirnames = os.listdir(clip_images_dir)
    for clip_dirname in clips_dirnames:

        tmp_output_dir = os.path.join(output_dir, clip_dirname)
        Path(tmp_output_dir).mkdir(parents=True, exist_ok=True)

        images_dir = os.path.join(clip_images_dir, clip_dirname)

        inputs.append({'images_dir':images_dir,
                       'output_dir':tmp_output_dir})


    pool = Pool(args.j)
    _ = list(
        tqdm(
            pool.imap_unordered(
                extract_frames_for_sfm, inputs),
                total=len(inputs)
        )
    )

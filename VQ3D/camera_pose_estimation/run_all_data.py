import os
import sys
import json
import argparse
import numpy as np
import subprocess as sp
from imageio import imread
from PIL import Image

from pathlib import Path

def load_vq3d_annotation(filename):
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        scan_uid = video['scan_uid']
        for clip in video['clips']:
            output[clip['clip_uid']] = scan_uid
    return output


#scan name to uid
scan_name_to_uid = {
    'unict_Scooter mechanic_31': 'unict_3dscan_001',
    'unict_Baker_32': 'unict_3dscan_002',
    'unict_Carpenter_33': 'unict_3dscan_003',
    'unict_Bike mechanic_34': 'unict_3dscan_004',
}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_from_videos_frames/',
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--query_filename",
        type=str,
        default='data/vq3d_train.json',
        help="Input query file.",
    )
    parser.add_argument(
        "--camera_intrinsics_filename",
        type=str,
        default='data/scan_to_intrinsics.json',
        help="Json file with all the camera intrinsics.",
    )
    parser.add_argument(
        "--scans_keypoints_dir",
        type=str,
        default='data/scans_keypoints/',
        help="Input folder with the scans keypoints.",
    )
    parser.add_argument(
        "--scans_dir",
        type=str,
        default='data/scans/',
        help="Input folder with the scans keypoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/clips_camera_poses/',
        help="Output folder.",
    )
    args = parser.parse_args()

    #ROOT DATAPATH
    input_dir=args.input_dir
    scans_dir=args.scans_dir
    scans_keypoints_dir=args.scans_keypoints_dir

    # output dir
    output_dir_root=args.output_dir
    Path(output_dir_root).mkdir(parents=True, exist_ok=True)

    # load ALL intrinsics
    all_intrinsics = json.load(open(args.camera_intrinsics_filename, 'r'))

    # load query file
    clip_uid_to_scan_uid = load_vq3d_annotation(args.query_filename)

    clip_uids = list(clip_uid_to_scan_uid.keys())

    for clip_uid in clip_uids:

        # check if not processed or not in process
        output_dir = os.path.join(output_dir_root, clip_uid)
        if os.path.isdir(output_dir): continue

        #create directories
        egovideo_dir = os.path.join(output_dir, 'egovideo')
        scan_dir = os.path.join(output_dir, 'scan')

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(egovideo_dir).mkdir(parents=True, exist_ok=True)
        Path(scan_dir).mkdir(parents=True, exist_ok=True)


        scan_name = clip_uid_to_scan_uid[clip_uid]
        scan_uid = scan_name_to_uid[scan_name]

        #scan-parameters
        os.symlink(os.path.join('../../../../',scans_keypoints_dir, scan_uid),
                   os.path.join(scan_dir, 'descriptors'))

        #scan data
        os.symlink(os.path.join('../../../../', scans_dir, scan_uid),
                   os.path.join(scan_dir, 'matterpak'))

        #symlink data
        # frames
        os.symlink(os.path.join('../../../../', input_dir, clip_uid),
                   os.path.join(egovideo_dir, 'color_distorted'))

        # resolution
        im = Image.open(os.path.join(egovideo_dir,
                                     'color_distorted',
                                     'color_0000000.jpg'))
        resolution = im.size # (width, height)
        resolution_token = (str(resolution[0]),
                            str(resolution[1]))
        resolution_token = str(resolution_token)
        im.close()

        # intrinsics
        intrinsics = all_intrinsics[scan_name][resolution_token]
        intr = []
        intr.append(str(resolution[0]))
        intr.append(str(resolution[1]))
        intr.append(str(intrinsics['cx']))
        intr.append(str(intrinsics['cy']))
        intr.append(str(intrinsics['f']))
        intr.append(str(intrinsics['k1']))
        intr.append(str(intrinsics['k2']))
        intr = ' '.join(intr)
        with open(os.path.join(egovideo_dir, 'fisheye_intrinsics.txt'), 'w') as f:
            f.write(intr)


        ######################################
        ######################################
        # command
        cmd = ["./main.sh",
               f"./{scan_dir}",
               f"./{egovideo_dir}",
              ]

        # get flags
        if not os.path.isfile(os.path.join(egovideo_dir,
                                           'superglue_track',
                                           'poses',
                                           'good_pose_reprojection.npy'
                                          )):
            cmd.append("--sfm")

        if not os.path.isfile(os.path.join(egovideo_dir,
                                           'superglue_track',
                                           'track.npy',
                                          )):
            cmd.append("--track")

        if not os.path.isfile(os.path.join(egovideo_dir,
                                           'poses_reloc',
                                           'camera_poses_pnp.npy',
                                          )):
            cmd.append("--pnp")

        if not os.path.isfile(os.path.join(egovideo_dir,
                                           'vlad_best_match',
                                           'queries.pkl',
                                          )):
            cmd.append("--database")

        if not os.path.isfile(os.path.join(egovideo_dir,
                                           'intrinsics.txt',
                                          )):
            cmd.append("--undistort")
        
        if not os.path.isdir(os.path.join(egovideo_dir,
                                          'pose_visualization_superglue',
                                          )):
            cmd.append("--viz")
        
        # start data processing
        o = sp.run(cmd)






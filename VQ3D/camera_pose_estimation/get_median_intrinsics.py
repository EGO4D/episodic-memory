import os
import sys
import json
import argparse
import numpy as np

sys.path.append('Camera_Intrinsics_API/')
from get_camera_intrinsics import CameraIntrinsicsHelper


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/videos_sfm/',
        help="COLMAP output folder of videos",
    )
    parser.add_argument(
        "--input_dir_greedy",
        type=str,
        default='data/videos_sfm_greedy/',
        help="Folder for the COLMAP outputs - greedy.",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default='data/v1/annotations/',
        help="annotation folder. Must contain the vq3d_<split>.json files.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default='data/v1/scan_to_intrinsics.json',
    )
    args = parser.parse_args()


    dataset = {}
    for split in ['train', 'val']:
        a = json.load(open(os.path.join(args.annotation_dir,
                                        f'vq3d_{split}.json'), 'r'))
        for video in a['videos']:
            video_uid=video['video_uid']
            scan_uid=video['scan_uid']
            dataset[video_uid]=scan_uid

    helper = CameraIntrinsicsHelper()
    
    datadir=args.input_dir
    datadir_2=args.input_dir_greedy
    cpt=0
    all_intrinsics = {}
    for video_uid in os.listdir(datadir):
        scan_uid=dataset[video_uid]
        intrinsic_txt = os.path.join(datadir,
                                     video_uid,
                                    'sparse',
                                    '0',
                                    'cameras.txt')
    
        if not os.path.isfile(intrinsic_txt):
            intrinsic_txt = os.path.join(datadir_2,
                                         video_uid,
                                        'sparse',
                                        '0',
                                        'cameras.txt')
    
            if not os.path.isfile(intrinsic_txt):
                cpt+=1
            else:
                intrinsics = helper.parse_colmap_intrinsics(intrinsic_txt)
                if scan_uid not in all_intrinsics:
                    all_intrinsics[scan_uid]={}
                token = (intrinsics['width'], intrinsics['height'])
                if token not in all_intrinsics[scan_uid]:
                    all_intrinsics[scan_uid][token] = []
    
                all_intrinsics[scan_uid][token].append(
                    (
                      intrinsics['f'],
                      intrinsics['cx'],
                      intrinsics['cy'],
                      intrinsics['k1'],
                      intrinsics['k2'],
                    )
                )
        else:
            intrinsics = helper.parse_colmap_intrinsics(intrinsic_txt)
            if scan_uid not in all_intrinsics:
                all_intrinsics[scan_uid]={}
            token = (intrinsics['width'], intrinsics['height'])
            if token not in all_intrinsics[scan_uid]:
                all_intrinsics[scan_uid][token] = []
    
            all_intrinsics[scan_uid][token].append(
                (
                  intrinsics['f'],
                  intrinsics['cx'],
                  intrinsics['cy'],
                  intrinsics['k1'],
                  intrinsics['k2'],
                )
            )
    
    outputs = {}
    for scan_uid, d in all_intrinsics.items():
        print(' ')
        print('Scan uid: ', scan_uid)
        outputs[scan_uid]={}
        for resolution, v in d.items():
            print('   -- resolution: ', resolution)
            resolution_str = str(resolution)
            outputs[scan_uid][resolution_str]={
                'f':  np.median([float(i[0]) for i in v]),
                'cx': np.median([float(i[1]) for i in v]),
                'cy': np.median([float(i[2]) for i in v]),
                'k1': np.median([float(i[3]) for i in v]),
                'k2': np.median([float(i[4]) for i in v]),
            }
            for i in v:
                print('   -- -- -- : ', i)
            print(' ')
            print('   -- -- -- : ',
                  outputs[scan_uid][resolution_str]['f'],
                  outputs[scan_uid][resolution_str]['cx'],
                  outputs[scan_uid][resolution_str]['cy'],
                  outputs[scan_uid][resolution_str]['k1'],
                  outputs[scan_uid][resolution_str]['k2'],
                 )
    
    json.dump(outputs, open(output_filename, 'w'))


import os
import sys
import csv
import json
import argparse
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth


def parse_VQ2D_queries(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        video_uid = video['video_uid']
        if video_uid not in output:
            output[video_uid]={}
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if clip_uid not in output[video_uid]:
                output[video_uid][clip_uid]={}
            for ai, annot in enumerate(clip['annotations']):
                if ai not in output[video_uid][clip_uid]:
                    output[video_uid][clip_uid][ai]={}
                for qset_id, qset in annot['query_sets'].items():
                    if not qset["is_valid"]:
                        continue
                    output[video_uid][clip_uid][ai][qset_id] = {
                        "query_frame": qset["query_frame"],
                        "object_title": qset["object_title"],
                    }
    return output


Rz_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0, 0],
                  [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                 ])



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_from_videos_camera_poses/',
        help="Camera pose folder"
    )
    parser.add_argument(
        "--vq3d_queries",
        type=str,
        default='data/vq3d_val.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default='data/vq3d_val_wgt.json',
        help="VQ3D query file with GT pred vector 3d"
    )
    parser.add_argument(
        "--vq2d_queries",
        type=str,
        default='data/vq_val.json',
        help="VQ3D query file"
    )

    args = parser.parse_args()

    root_dir = args.input_dir
    output_filename = args.output_filename

    # Visual Query 3D queries
    vq3d_queries = json.load(open(args.vq3d_queries, 'r'))

    # Visual Query 2D results
    vq2d_queries = parse_VQ2D_queries(args.vq2d_queries)

    # Load mapping VQ2D to VQ3D queries/annotations
    if 'val' in args.vq2d_queries:
        split='val'
    elif 'train' in args.vq2d_queries:
        split='train'
    elif 'test' in args.vq2d_queries:
        split='test'
    else:
        raise ValueError
    query_matching_filename=f'data/mapping_vq2d_to_vq3d_queries_annotations_{split}.json'
    query_matching = json.load(open(query_matching_filename, 'r'))

    helper = VisualQuery3DGroundTruth()

    cpt_valid_queries = 0
    for video in vq3d_queries['videos']:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                for qset_id, qset in annot['query_sets'].items():

                    mapping_ai=query_matching[video_uid][clip_uid][str(ai)][qset_id]['ai']
                    mapping_qset_id=query_matching[video_uid][clip_uid][str(ai)][qset_id]['qset_id']

                    assert qset['object_title']==vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['object_title']
                    query_frame=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['query_frame']

                    # -- -- get GT object centroid in world system
                    for w in [1, 2]:
                        vec = helper.load_3d_annotation(qset[f'3d_annotation_{w}'])
                        
                        vec = np.append(vec, 1.)
                        vec = np.matmul(Rz_90, vec)
                        vec = vec[:3] / vec[3]

                        qset[f'gt_3d_vec_world_{w}'] = vec.tolist()


                    # -- -- compute vector in Query frame coord system
                    # get pose
                    dirname = os.path.join(root_dir,
                                           clip_uid,
                                           'egovideo')

                    if not os.path.isdir(dirname): continue

                    poses = helper.load_pose(dirname)

                    if poses is None: continue

                    T, valid_pose = poses

                    if query_frame > (len(valid_pose) - 1):
                        continue
                    if not valid_pose[query_frame]:
                        continue

                    pose = T[query_frame]

                    pose_inv = np.linalg.inv(pose)

                    for w in [1, 2]:
                        vec = np.array(qset[f'gt_3d_vec_world_{w}'])
                        vec = np.append(vec, 1.0)

                        gt_3d_vec = np.matmul(pose_inv, vec)
                        gt_3d_vec = gt_3d_vec[:3] / gt_3d_vec[3]

                        qset[f'gt_3d_vec_{w}'] = gt_3d_vec.tolist()

                    cpt_valid_queries += 1

print('number of valide queries: ', cpt_valid_queries)
json.dump(vq3d_queries, open(output_filename, 'w'))

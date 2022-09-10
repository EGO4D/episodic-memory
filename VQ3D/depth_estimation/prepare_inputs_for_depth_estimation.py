import os
import shutil
import sys
import json
import torch
import h5py
import copy
import argparse
import numpy as np
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

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
                        "response_track": qset["response_track"],
                    }
    return output

def parse_VQ3D_queries(filename: str) -> List:
    data = json.load(open(filename, 'r'))
    clips = []
    for video in data['videos']:
        for clip in video['clips']:
            clips.append(clip['clip_uid'])
    return clips


def parse_VQ2D_predictions(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for i in range(len(data['dataset_uids'])):
        dataset_uid = data['dataset_uids'][i]
        output[dataset_uid] = {'pred': data['predicted_response_track'][i],
                               'gt': data['ground_truth_response_track'][i]}
    return output


r"""
list[
    {'metadata':{
        'video_uid'
        'video_start_sec'
        'video_end_sec'
        'clip_fps'
    }
     'clip_uid',
     'query_set',
     'query_frame',
     'response_track',
     'visual_crop',
     'object_title',
     'dataset_uid'}
    ]
"""
def parse_VQ2D_mapper(filename: str) -> Dict:
    data = json.load(open(filename,'r'))
    output = {}
    for i in range(len(data)):
        dataset_uid = data[i]['dataset_uid']
        video_uid = data[i]['metadata']['video_uid']
        clip_uid = data[i]['clip_uid']
        query_set = data[i]['query_set']
        query_frame = data[i]['query_frame']
        object_title = data[i]['object_title']
        visual_crop = data[i]['visual_crop']
        if video_uid not in output:
            output[video_uid]={}
        if clip_uid not in output[video_uid]:
            output[video_uid][clip_uid]={}
        if query_set not in output[video_uid][clip_uid]:
            output[video_uid][clip_uid][query_set]={}
        if query_frame not in output[video_uid][clip_uid][query_set]:
            output[video_uid][clip_uid][query_set][query_frame]=[]

        output[video_uid][clip_uid][query_set][query_frame].append(
            {'dataset_uid':dataset_uid,
             'object_title':object_title,
             'visual_crop':visual_crop,
            }
        )
    return output



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_from_videos_camera_poses/',
        help="Camera pose folder"
    )
    parser.add_argument(
        "--vq2d_results",
        type=str,
        default='data/vq2d_results/siam_rcnn_residual_kys_val.json',
        help="filename for the VQ2D results"
    )
    parser.add_argument(
        "--vq2d_annot",
        type=str,
        default='data/val_annot.json',
        help="VQ2D mapping queries"
    )
    parser.add_argument(
        "--vq3d_queries",
        type=str,
        default='data/vq3d_val.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--vq2d_queries",
        type=str,
        default='data/vq_val.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--use_gt",
        action='store_true'
    )
    args = parser.parse_args()

    root_dir = args.input_dir

    # Visual Query 3D queries
    vq3d_queries = json.load(open(args.vq3d_queries, 'r'))

    # Visual Query 2D results
    vq2d_queries = parse_VQ2D_queries(args.vq2d_queries)
    vq2d_pred = parse_VQ2D_predictions(args.vq2d_results)
    vq2d_mapping = parse_VQ2D_mapper(args.vq2d_annot)

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

                    dataset_uid=vq2d_mapping[video_uid][clip_uid][mapping_qset_id][query_frame][0]['dataset_uid']

                    if args.use_gt:
                        response_track = vq2d_pred[dataset_uid]['gt']
                        response_track=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["response_track"]
                        frame_indices=[x['frame_number'] for x in response_track]
                    else:
                        response_track = vq2d_pred[dataset_uid]['pred'][0]
                        frames = response_track['bboxes']
                        frame_indices = [x['fno'] for x in frames]


                    depth_dir = os.path.join(root_dir,
                                             clip_uid,
                                             'egovideo',
                                             'depth_DPT_predRT'
                                            )
                    Path(depth_dir).mkdir(parents=True, exist_ok=True)

                    tmp_input_depth_dir = os.path.join(root_dir,
                                                       clip_uid,
                                                       'egovideo',
                                                       'input_depth_DPT_predRT',
                                                      )
                    Path(tmp_input_depth_dir).mkdir(parents=True, exist_ok=True)

                    for frame_index in frame_indices:
                        framename = 'color_%07d' % frame_index
                        src = os.path.join(root_dir,
                                                 clip_uid,
                                                 'egovideo',
                                                 'color',
                                                 framename+'.jpg')
                        if os.path.isfile(src):
                            shutil.copy(src,
                                        os.path.join(tmp_input_depth_dir,
                                                     framename+'.jpg')
                                       )


import os
import sys
import json
import h5py
import torch
import argparse
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth


def scale_im_height(image, H):
    im_H, im_W = image.shape[:2]
    W = int(1.0 * H * im_W / im_H)
    return cv2.resize(image, (W, H))

def _get_box(annot_box):
    x, y, w, h = annot_box["x"], annot_box["y"], annot_box["width"], annot_box["height"]
    return (int(x), int(y), int(x + w), int(y + h))

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
                        "visual_crop": qset["visual_crop"],
                        "response_track": qset["response_track"],
                    }
    return output

def parse_VQ2D_predictions(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for i in range(len(data['dataset_uids'])):
        dataset_uid = data['dataset_uids'][i]
        output[dataset_uid] = {'pred': data['predicted_response_track'][i],
                               'gt': data['ground_truth_response_track'][i]}
    return output


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
        "--output_filename",
        type=str,
        default='data/vq3d_results/siam_rcnn_residual_kys_val.json',
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
        default='data/vq3d_val_wgt.json',
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
    parser.add_argument(
        "--use_depth_from_scan",
        action='store_true'
    )
    args = parser.parse_args()

    root_dir = args.input_dir

    output_filename = args.output_filename

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

                    object_title=qset['object_title']
                    assert qset['object_title']==vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['object_title']
                    query_frame=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['query_frame']
                    oW=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["visual_crop"]["original_width"]
                    oH=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["visual_crop"]["original_height"]

                    dataset_uid=vq2d_mapping[video_uid][clip_uid][qset_id][query_frame][0]['dataset_uid']
                    #dataset_uid=vq2d_mapping[video_uid][clip_uid][mapping_qset_id][query_frame][0]['dataset_uid']

                    # get intrinsics
                    camera_intrinsics = np.loadtxt(os.path.join(root_dir,
                                                                clip_uid,
                                                                'egovideo',
                                                                'fisheye_intrinsics.txt'))
                    W = camera_intrinsics[0]
                    H = camera_intrinsics[1]
                    f = camera_intrinsics[4]
                    k1 = camera_intrinsics[5]
                    k2 = camera_intrinsics[6]
                    cx = W/2.0
                    cy = H/2.0



                    # get poses
                    dirname = os.path.join(root_dir, clip_uid, 'egovideo')

                    if not os.path.isdir(dirname): continue

                    poses = helper.load_pose(dirname)
                    if poses is None: continue

                    T, valid_pose = poses

                    frame_indices_valid = []
                    local_frame_indices = []

                    # get RT frames with poses
                    if args.use_gt:
                        response_track=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["response_track"]
                        for i, frame in enumerate(response_track):
                            frame_index = frame['frame_number']

                            if (frame_index > -1) and (frame_index < len(valid_pose)):

                                box = _get_box(frame)
                                x1, y1, x2, y2 = box

                                if (x1<(W-1)) and (x2>1) and (y1<(H-1)) and (y2>1):
                                    # check if pose is valid
                                    if valid_pose[frame_index]:
                                        frame_indices_valid.append(frame_index)
                                        local_frame_indices.append(i)

                    else:
                        response_track = vq2d_pred[dataset_uid]['pred'][0]

                        frames = response_track['bboxes']

                        frame_indices = [x['fno'] for x in frames]

                        for i, frame_index in enumerate(frame_indices):

                            # check if frame index is valid
                            if (frame_index > -1) and (frame_index < len(valid_pose)):

                                # check if box is within frame bound:
                                box = frames[i]
                                x1 = box['x1']
                                x2 = box['x2']
                                y1 = box['y1']
                                y2 = box['y2']

                                if (x1<(W-1)) and (x2>1) and (y1<(H-1)) and (y2>1):

                                    # check if pose is valid
                                    if valid_pose[frame_index]:
                                        frame_indices_valid.append(frame_index)
                                        local_frame_indices.append(i)

                    if len(frame_indices_valid) == 0: continue


                    # get the last frame of the RT
                    j = np.argmax(frame_indices_valid)
                    frame_index_valid = frame_indices_valid[j]
                    local_frame_index = local_frame_indices[j]


                    # check if Query frame has pose
                    if valid_pose[query_frame]:
                        pose_Q = T[query_frame]
                    else:
                        pose_Q = None


                    # get RT frame pose
                    pose = T[frame_index_valid]

                    cpt_valid_queries+=1


                    # get depth
                    if args.use_depth_from_scan:
                        depth_dir = os.path.join(root_dir,
                                                clip_uid,
                                                'egovideo',
                                                'pose_visualization_depth_superglue'
                                                )
                        framename = 'render_%07d' % frame_index_valid
                        depth_filename = os.path.join(depth_dir,
                                                      framename+'.h5')
                        if os.path.isfile(depth_filename):
                            data = h5py.File(depth_filename)
                            depth = np.array(data['depth']) # in meters
                        else:
                            print('missing predicted depth')
                            continue
                    else:
                        depth_dir = os.path.join(root_dir,
                                                clip_uid,
                                                'egovideo',
                                                'depth_DPT_predRT'
                                                )
                        framename = 'color_%07d' % frame_index_valid
                        depth_filename = os.path.join(depth_dir,
                                                      framename+'.pfm')
                        if os.path.isfile(depth_filename):
                            data, scale = helper.read_pfm(depth_filename)
                        else:
                            print('missing predicted depth')
                            continue

                        depth = data/1000.0 # in meters


                    # resize depth
                    depth = torch.FloatTensor(depth)
                    depth = depth.unsqueeze(0).unsqueeze(0)
                    if args.use_gt:
                        depth = torch.nn.functional.interpolate(depth,
                                                                size=(int(oH),
                                                                      int(oW)),
                                                                mode='bilinear',
                                                                align_corners=True)
                    else:
                        depth = torch.nn.functional.interpolate(depth,
                                                                size=(int(H), int(W)),
                                                                mode='bilinear',
                                                                align_corners=True)
                    depth = depth[0][0].cpu().numpy()

                    # select d
                    if args.use_gt:
                        box = _get_box(response_track[local_frame_index])
                        x1, y1, x2, y2 = box
                    else:
                        box = frames[local_frame_index]
                        x1 = box['x1']
                        x2 = box['x2']
                        y1 = box['y1']
                        y2 = box['y2']
                        if x1<0: x1=0
                        if y1<0: y1=0

                    d = depth[y1:y2, x1:x2]

                    if d.size == 0:
                        continue

                    d = np.median(d)

                    tx = (x1+x2)/2.0
                    ty = (y1+y2)/2.0

                    # vec in current frame:
                    z = d
                    x = z * (tx -cx -0.5)/f
                    y = z * (ty -cy -0.5)/f
                    vec = np.ones(4)
                    vec[0]=x
                    vec[1]=y
                    vec[2]=z

                    # object center in world coord system
                    pred_t = np.matmul(pose, vec)
                    pred_t = pred_t / pred_t[3]

                    print('Clip_uid: ', clip_uid,
                          ' - ai: ', ai,
                          ' - qset_id: ', qset_id,
                          ' - object_title: ', object_title)
                    # object center in Query frame coord system
                    if pose_Q is not None:
                        vec = np.matmul(np.linalg.inv(pose_Q), pred_t)
                        vec = vec / vec[3]
                        vec = vec[:3]
                        qset['pred_3d_vec'] = vec.tolist()
                        l1 = np.linalg.norm(vec-qset['gt_3d_vec_1'])
                        l2 = np.linalg.norm(vec-qset['gt_3d_vec_2'])
                        print('L2 distance with annotation 1 and 2 in query frame coord system',
                              l1, ' ', l2)
                    else:
                        vec = None

                    pred_t = pred_t[:3]
                    world_dist_1 = np.linalg.norm(pred_t-qset['gt_3d_vec_world_1'])
                    world_dist_2 = np.linalg.norm(pred_t-qset['gt_3d_vec_world_2'])
                    print('L2 distance with annotation 1 and 2 in world coord system',
                          world_dist_1, ' ', world_dist_2, '\n')

                    # save output for metric compute
                    qset['pred_3d_vec_world'] = pred_t.tolist()

    print(' valide # queries: ', cpt_valid_queries)
    json.dump(vq3d_queries, open(output_filename, 'w'))


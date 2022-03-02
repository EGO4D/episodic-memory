import os
import sys
import cv2
import json
import fnmatch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

import torch
from torch.utils.data.dataset import Dataset

from utils import *

sys.path.append('./SuperGlueMatching')
from models.utils import read_image
from models.superpoint import SuperPoint
from models.superglue import SuperGlue



def SuperGlueMatcher(model,
                     superpoints_0,
                     superpoints_1):

    data = {
        'descriptors0': superpoints_0['descriptors'][0].unsqueeze(0).cuda(),
        'descriptors1': superpoints_1['descriptors'][0].unsqueeze(0).cuda(),
        'keypoints0': superpoints_0['keypoints'][0].unsqueeze(0).cuda(),
        'keypoints1': superpoints_1['keypoints'][0].unsqueeze(0).cuda(),
        'scores0': superpoints_0['scores'][0].unsqueeze(0).cuda(),
        'scores1': superpoints_1['scores'][0].unsqueeze(0).cuda(),
        'image0': torch.zeros((1,1,480,640)),
        'image1': torch.zeros((1,1,480,640)),
    }
    match = model(data)

    confidence = match['matching_scores0'][0].detach().cpu().numpy()
    matches = match['matches0'][0].cpu().numpy()
    kpts0 = superpoints_0['keypoints'][0].cpu().numpy()
    kpts1 = superpoints_1['keypoints'][0].cpu().numpy()

    valid = matches > -1

    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    ind0 = np.nonzero(valid)[0]
    ind1 = matches[valid]

    confidence = confidence[valid]

    return mkpts0, mkpts1, ind0, ind1, confidence





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract and track keypoints throughout the video using SuperGlue.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--extract_descriptor', action='store_true')
    parser.add_argument(
        '--ego_dataset_folder', type=str,
        help='Ego dataset folder')

    opt = parser.parse_args()
    print(opt)

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    superpoint = SuperPoint(config.get('superpoint', {})).cuda()
    superpoint = superpoint.eval()

    superglue = SuperGlue(config.get('superglue', {})).cuda()
    superglue = superglue.eval()

    ROOT_DIR = opt.ego_dataset_folder
    OUTPUT_DIR = 'superglue_track'

    if not os.path.exists(os.path.join(ROOT_DIR, OUTPUT_DIR)):
        os.mkdir(os.path.join(ROOT_DIR, OUTPUT_DIR))

    # EXTRACT KPTS AND DESCRIPTORS, RUN ONCE!
    if opt.extract_descriptor:
        loc_list = []
        des_list = []
        superpoints_list = []

        all_images = np.arange(0, len(fnmatch.filter(os.listdir(os.path.join(ROOT_DIR, 'color')), '*.jpg')), step=1)

        for index in tqdm(all_images):
            color_info = os.path.join(ROOT_DIR, 'color/color_%07d.jpg' % index)
            _, gray_tensor, scale = read_image(color_info, 'cpu', resize=[640, 480], rotation=0, resize_float=False)
            gray_tensor = gray_tensor.reshape(1, 1, 480, 640)
            input_batch = {'image': gray_tensor.cuda()}

            with torch.no_grad():
                output = superpoint(input_batch)

            output_ = {k: [output[k][i].detach().cpu()\
                           for i in range(len(output[k]))]\
                       for k in output}

            output_['scale']=scale

            superpoints_list.append(output_)

            output_np = {k: [output[k][i].detach().cpu().numpy()\
                             for i in range(len(output[k]))]\
                         for k in output}

            des = np.asarray(output_np['descriptors'][0])
            loc = np.asarray(
                [output_np['keypoints'][0][j] * scale\
                 for j in range(output_np['keypoints'][0].shape[0])
                ]
            )

            loc_list.append(loc)
            des_list.append(des.transpose())

        torch.save(superpoints_list, '%s/superpoints.pkl' % os.path.join(ROOT_DIR, OUTPUT_DIR))
        np.savez('%s/keypoints.npz' % os.path.join(ROOT_DIR, OUTPUT_DIR), loc_list)
        np.savez('%s/descriptors.npz' % os.path.join(ROOT_DIR, OUTPUT_DIR), des_list)

    else:
        # -- load superpoint output
        superpoint_filename=os.path.join(ROOT_DIR, OUTPUT_DIR, 'superpoints.pkl')
        assert os.path.isfile(superpoint_filename)
        superpoints_list=torch.load(superpoint_filename)

    superglue_conf_thresh=0.9
    num_images = len(superpoints_list)
    print('num image: ', num_images)
    K = np.loadtxt('%s/intrinsics.txt' % ROOT_DIR)

    original_image_id_list = np.arange(0,
                                       len(fnmatch.filter(os.listdir(opt.ego_dataset_folder + '/color/'),
                                                          '*.jpg')),
                                       step=1)

    assert num_images == len(original_image_id_list)

    def get_tracks_parallel(inputs):
        i = inputs['i']
        superpoints_list = inputs['superpoints_list']
        K = inputs['K']

        loc = superpoints_list[i]['keypoints'][0].numpy()

        num_points = loc.shape[0]
        num_images = len(superpoints_list)

        track_i = {}
        for j in range(i + 1, min(num_images, i + 5)):
            # Match features between the i-th and j-th images
            x1, x2, ind1, ind2, conf = SuperGlueMatcher(
                superglue,
                superpoints_list[i],
                superpoints_list[j]
            )

            m = conf > superglue_conf_thresh
            x1 = x1[m]
            x2 = x2[m]
            ind1 = ind1[m].astype(int)
            ind2 = ind2[m].astype(int)

            num_points_j = superpoints_list[j]['keypoints'][0].shape[0]
            track_i[j] = -np.ones(num_points_j, dtype=int)
            track_i[j][ind2] = ind1

        return (i, track_i)

    inputs = [{'i':i,
               'superpoints_list':superpoints_list.copy(),
               'K':K.copy()} for i in range(num_images - 1)]

    # build track dictionary
    track_dict = {}
    for x in tqdm(inputs):
        r = get_tracks_parallel(x)

        if r is None: continue

        i, track_i = r
        # merge the current track dict with the main one
        for j, v in track_i.items():
            if j not in track_dict:
                track_dict[j] = {}

            assert i not in track_dict[j]

            track_dict[j][i] = v.tolist()

    json.dump(track_dict,
              open(os.path.join(
                                ROOT_DIR,
                                OUTPUT_DIR,
                                'track_dict.json'
                                ),
                'w')
    )

    # build track matrix
    # NOTE: This will potentially generate duplicates
    # this is just redondant information for the 3D Triangulation
    def fetch_graph(g, i, f_idx, start_i):
        # prevent from matching features more than 20frames apart
        if (start_i-i) > 20:
            return []
        if i not in g:
            return []
        output = []
        for j, v in g[i].items():
            if v[f_idx] > -1:
                output.append((i, f_idx))
                for x in fetch_graph(g, j, v[f_idx], start_i):
                    output.append(x)
                # only one match per image for each feature
                break
        return output

    inputs = [
        {'i':i,
         'track_dict': track_dict.copy(),
         'superpoints_list': superpoints_list.copy(),
         'K': K.copy(),
        } for i in range(num_images, 0, -1)
    ]

    track = []
    for x in tqdm(inputs):
        i = x['i']
        track_dict = x['track_dict']
        superpoints_list = x['superpoints_list']
        K = x['K']
        if i in track_dict:
            scale = superpoints_list[i]['scale']
            for f_idx in range(superpoints_list[i]['keypoints'][0].shape[0]):

                tmp_track = -np.ones((num_images,2))
                mask = np.zeros(num_images)
                tracklets = fetch_graph(track_dict, i, f_idx, i)

                if not tracklets: continue

                for tr in tracklets:

                    if not tr: continue

                    k, tmp_f_idx = tr

                    tmp_track[k,:]=superpoints_list[k]['keypoints'][0][tmp_f_idx].cpu().numpy().copy() * scale
                    mask[k] = 1

                count = np.sum(mask)
                if count > 3:
                    mask = mask.astype(bool)

                    # normalize indices
                    tmp_track_n = np.hstack([tmp_track,
                                             np.ones((tmp_track.shape[0], 1))]) @ np.linalg.inv(K).T
                    tmp_track_n = tmp_track_n[:, :2]
                    tmp_track_n[~mask] = -1000

                    tmp_track_n = np.expand_dims(tmp_track_n, axis=1)

                    track.append(tmp_track_n)

    track = np.concatenate(track, axis=1)

    np.save('%s/track.npy' % os.path.join(ROOT_DIR, OUTPUT_DIR), track)
    np.save('%s/original_image_id.npy' % os.path.join(ROOT_DIR, OUTPUT_DIR), original_image_id_list)
    print('total number of feature: ', track.shape[1])

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
import h5py
import pickle
import torch.nn.functional as F
from scipy.io import loadmat

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.input_feat_dim = opt['input_feat_dim']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.gap = opt['stitch_gap']
        self.clip_anno = opt['clip_anno']
        self.moment_classes = opt["moment_classes"]


        self._getDatasetDict()

        self.anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]

    def _getDatasetDict(self):

        anno_database = load_json(self.clip_anno)

        self.clip_dict = {}
        class_list = []

        for clip_name, clip_info in anno_database.items():

            clip_subset = clip_info['subset']

            if clip_subset in self.subset:
                self.clip_dict[clip_name] = clip_info

                for item in clip_info['annotations']:
                    class_list.append(item['label'])

        self.clip_list = list(self.clip_dict.keys())

        if os.path.exists(self.moment_classes):
            with open(self.moment_classes, 'r') as f:
                self.classes = json.load(f)
        else:
            class_list = list(set(class_list))
            class_list = sorted(class_list)
            print(f'The total number of classes is {len(class_list) + 1}')
            self.classes = {'Background': 0}
            for i,cls in enumerate(class_list):
                self.classes[cls] = i + 1
            with open(self.moment_classes, 'w') as f:
                f.write(json.dumps(self.classes))

    def __getitem__(self, index):
        if self.mode == "train":
            video_data, match_score_action, match_score_start, match_score_end, gt_bbox, num_gt, num_frms = self._get_video_data(index)
            return video_data, match_score_action, match_score_start, match_score_end, gt_bbox, num_gt, num_frms
        else:
            video_data, num_frms = self._get_video_data(index)
            return index, video_data, num_frms



    def _get_video_data(self, index):

        # General data
        clip_name = self.clip_list[index]
        clip_info = self.clip_dict[clip_name]
        video_name = clip_info['video_id']

        # Get video features
        v_data = torch.load(os.path.join(self.feature_path, video_name + '.pt'))
        v_data = torch.transpose(v_data, 0, 1)
        num_frms_v = v_data.shape[-1]
        fps_v = clip_info['fps']

        clip_start = int(clip_info['parent_start_sec'] * fps_v)
        clip_end = min(int(clip_info['parent_end_sec'] * fps_v), num_frms_v-1)

        video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)
        win_data = v_data[:, clip_start: clip_end+1]
        num_frms = min(win_data.shape[-1], self.temporal_scale)
        video_data[:, :num_frms] = win_data[:, :num_frms]
        if self.mode == 'train':
            match_score_action, match_score_start, match_score_end, gt_bbox_padding, num_gt, num_frms = \
                self._get_train_data_label_org(num_frms, clip_name, fps_v)
            return video_data, match_score_action, match_score_start, match_score_end, gt_bbox_padding, num_gt, num_frms
        else:
            return video_data, num_frms



    def _get_train_data_label_org(self, num_frms, clip_name,  fps):
        # Get annotations
        clip_info = self.clip_dict[clip_name]
        clip_labels = clip_info['annotations']

        # Get gt_iou_map
        gt_bbox = []
        for j in range(len(clip_labels)):
            tmp_info = clip_labels[j]

            tmp_start_f = max(min(num_frms-1, tmp_info['start_time']*fps), 0)
            tmp_end_f = max(min(num_frms-1, tmp_info['end_time']*fps), 0)

            tmp_start = tmp_start_f / self.temporal_scale
            tmp_end = tmp_end_f / self.temporal_scale


            print(f'tmp_start {tmp_start}')
            print(f'tmp_end {tmp_end}')
            tmp_class = self.classes[tmp_info['label']]
            gt_bbox.append([tmp_start, tmp_end, tmp_class])


        # Get actionness scores
        match_score_action = [0] * self.temporal_scale
        for bbox in gt_bbox:
            left_frm = max(round(bbox[0] * self.temporal_scale), 0)
            right_frm = min(round(bbox[1] * self.temporal_scale), self.temporal_scale-1)
            match_score_action[left_frm:right_frm+1] = [bbox[2]] * (right_frm + 1 - left_frm)

        match_score_action = torch.Tensor(match_score_action)

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        if gt_bbox.shape[0] == 0:
            print(gt_bbox.shape)

        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_len_small = 3 * self.temporal_gap
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        max_num_box = 50
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_bbox_padding = gt_bbox.new(max_num_box, gt_bbox.size(1)).zero_()
        num_gt = min(gt_bbox.size(0), max_num_box)
        gt_bbox_padding[:num_gt, :] = gt_bbox[:num_gt]
        # labels = BoxList(torch.Tensor(gt_bbox))

        return  match_score_action, match_score_start, match_score_end, gt_bbox_padding, num_gt, num_frms



    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __len__(self):
        return len(self.clip_list)


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard



""" Dataset loader for the Ego4D language dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import glob
import numpy as np
import pandas as pd

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

from transformers import BertTokenizer, BertModel


class Ego4DClip(data.Dataset):
    def __init__(self, split, temp=None):
        super(Ego4DClip, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.window = config.DATASET.WINDOW
        self.min_duration = 3
        stride = int(self.window / 2)  # windos overlap by half

        self.debug = config.DEBUG
        self.temp = temp

        # load annotations
        if split == "train":
            anno_path = os.path.join(self.data_dir, "nlq_train.json")
        elif split == "test":  # use val set for test
            anno_path = os.path.join(self.data_dir, "nlq_val.json")
        with open(anno_path) as f:
            anno_json = json.load(f)
        ## - load csv from data/Ego4D_clip/benchmarck_lq_val_annotation.csv
        # anno_df = pd.read_csv(
        #     # os.path.join(self.data_dir, 'benchmarck_lq_clip_{}_v3_official_subsetv1.csv'.format(split)),
        #     os.path.join(self.data_dir, 'benchmarck_lq_clip_{}_v3_official_subsetv1.csv'.format(split)),
        #     delimiter=', ')

        # with open(os.path.join(self.data_dir, 'nlq_text_features.npy'), 'rb') as f:
        #     self.bert_feature = torch.load(f)
        ## - random feature experiment
        # for k,v in self.bert_feature.items():
        #     pass
        #     # v.share_memory()
        #     # self.bert_feature[k] = torch.randn_like(v)

        ## - load template
        # with open(os.path.join(self.data_dir, 'ego4d_nlq_query_template_v3.json'), 'rb') as f:
        #     self.query_template = json.load(f)

        # with open(self.data_dir + '/action/uid_to_duration.json', 'r') as f:
        #     self.video_duration = json.load(f)
        #     print('load duration of {} videos'.format(len(list(self.video_duration.keys()))))

        anno_pairs = []
        clip_missing = [
            "c36ac819-ac4c-4550-abb1-625b9933a024",
            "7a2ceacd-22e6-497d-a035-986af56b5c57",
            "2e25aa7b-d3cd-458f-8011-dcc91f5da18c",
            "95a12e1e-e860-4ef9-a95c-2f317144f4c3",
            "6ec147ba-3180-48ea-92aa-472cad8ab2c5",
            "2332d8c4-ac84-4512-8109-58c61d7bce9b",
        ]

        for anno_video in anno_json["videos"]:
            for anno_clip in anno_video["clips"]:
                video_name = anno_clip["clip_uid"]
                if video_name in clip_missing:
                    continue
                clip_times = float(anno_clip["clip_start_sec"]), float(
                    anno_clip["clip_end_sec"]
                )  #
                clip_duration = clip_times[1] - clip_times[0]
                # - enumerate annotations
                for anno in anno_clip["annotations"]:
                    for query in anno["language_queries"]:
                        query_times = float(query["start_sec"]), float(query["end_sec"])
                        query_duration = (
                            query_times[1] - query_times[0]
                        )  # in terms of seconds

                        if split == "train":  # split to windows
                            if self.min_duration < query_duration < self.window and (
                                not self.debug or i < 200
                            ):
                                # find the new start and end time in the window
                                first_window_start = (
                                    np.ceil((query_times[1] - self.window) / stride)
                                    * stride
                                )
                                last_window_start = query_times[0] // stride * stride
                                # use negative windows for none-action clips
                                if config.DATASET.NEGWINDOW:
                                    first_window_start += -stride * 2
                                    last_window_start += stride * 2
                                first_window_start = max(0, int(first_window_start))
                                last_window_start = max(0, int(last_window_start))
                                # add all windows to annotations
                                for w_start in range(
                                    first_window_start,
                                    last_window_start + stride,
                                    stride,
                                ):
                                    if w_start + self.window > clip_duration + stride:
                                        continue
                                    new_anno = {
                                        "video": video_name,
                                        "description": query["query"],
                                        "window": [w_start, w_start + self.window],
                                        "clip_duration": clip_duration,
                                        "times": [
                                            query_times[0] - w_start,
                                            query_times[1] - w_start,
                                        ],
                                        # 'anno_uid': anno['anno_uid'].values[i],
                                        # 'query_idx': anno['query_idx'].values[i],
                                    }
                                    if w_start < clip_duration:
                                        anno_pairs.append(new_anno)

                        else:  # for val/test set, we need to process all windows
                            if self.min_duration < query_duration < self.window and (
                                not self.debug or i < 200
                            ):
                                for w_start in range(
                                    0, int(clip_duration) - self.window + stride, stride
                                ):
                                    new_anno = {
                                        "video": video_name,
                                        "description": query["query"],
                                        "window": [w_start, w_start + self.window],
                                        "clip_duration": clip_duration,
                                        "times": [query_times[0], query_times[1]],
                                        # 'anno_uid': anno_df['anno_uid'].values[i],
                                        # 'query_idx': anno_df['query_idx'].values[i],
                                    }
                                    if (
                                        self.temp is None
                                        or anno_df["query"].values[i]
                                        in self.query_template[self.temp]
                                    ):
                                        anno_pairs.append(new_anno)

        print(
            " -- collected {} samples for dataset {}".format(
                len(anno_pairs), (split, self.temp)
            )
        )
        self.annotations = anno_pairs
        self.cache_bert_feature = dict()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").cuda()

    def __getitem__(self, index):
        video_id = self.annotations[index]["video"]
        video_duration = self.annotations[index]["clip_duration"]

        # -- get start and end time, they should be related to the window
        gt_s_time, gt_e_time = self.annotations[index]["times"]
        sentence = self.annotations[index]["description"]
        window_se = self.annotations[index]["window"]

        window_start = window_se[0]
        window_end = window_se[1]
        duration = window_end - window_start

        # -- we use bert embedding
        if sentence not in self.cache_bert_feature:
            bert_feature = self.cache_text_feature(sentence)
            self.cache_bert_feature[sentence] = bert_feature
        word_vectors = self.cache_bert_feature[sentence]

        visual_input, visual_mask = self.get_video_features(
            video_id, video_duration, [window_start, window_end]
        )

        # -- Time scaled to same size
        assert config.DATASET.NUM_SAMPLE_CLIPS > 0
        try:
            visual_input = average_to_fixed_length(visual_input)
        except Exception as e:
            print("visual_input:", visual_input.shape, visual_input)
            print(e)
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
        overlaps = iou(
            torch.stack(
                [
                    s_times[:, None].expand(-1, num_clips),
                    e_times[None, :].expand(num_clips, -1),
                ],
                dim=2,
            )
            .view(-1, 2)
            .tolist(),
            torch.tensor([gt_s_time, gt_e_time]).tolist(),
        ).reshape(num_clips, num_clips)

        item = {
            "visual_input": visual_input,
            "vis_mask": visual_mask,
            "anno_idx": index,
            "word_vectors": word_vectors,
            "duration": self.window,
            "txt_mask": word_vectors[:, 0] != 0,
            "map_gt": torch.from_numpy(overlaps),
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def cache_text_feature(self, sentence):
        sentence = sentence.lower().strip("? \n.") + "?"
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {key: val.cuda() for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        output = outputs[0].data.detach()
        # print('given input {}, get {}'.format(inputs,output.shape))
        return output[0]  # dim:768

    def get_video_features(self, vid, duration, window_se=None):
        if "slowfast" == config.DATASET.VIS_INPUT_TYPE:
            # fps = 1
            feature = np.load(self.data_dir + "/sf/{}.npy".format(vid))

            # features = np.load(self.data_dir + '/2d/{}.npy'.format(vid))
            features = torch.tensor(feature).float()
            # duration = features.shape[0]*fps
            fps = 1.0 * features.shape[0] / duration

        else:
            raise NotImplementedError()

        """
        elif '3d' in config.DATASET.VIS_INPUT_TYPE:
            fps = 1.5
            feature_f = self.data_dir + '/3d/{}-*.npy'.format(vid)
            n_feat = len(glob.glob(feature_f))
            feature_list = []
            for i in range(n_feat):
                feature = np.load(self.data_dir + '/3d/{}-{}.npy'.format(vid,i))
                feature_list.append(torch.tensor(feature).float())
            features = torch.cat(feature_list)
        """
        ## - cut the video into window
        if window_se is not None:
            feat_start = int(window_se[0] * fps)
            feat_end = int(window_se[1] * fps)
            window_feature = features[feat_start:feat_end, :]
            assert (
                window_feature.shape[0] > 0
            ), "meet window feat se {} for video {}: {}, {} in duration: {}".format(
                (feat_start, feat_end), vid, features.shape, duration, window_se
            )
            features = window_feature
            # print(features.shape)

        # - normalize feature
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

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
        self.min_duration = 0 if split=='test' else 3 # discard moments less than 3 seconds
        stride = int(self.window / 2)  # windows overlap by half

        self.debug = config.DEBUG
        self.temp = temp

        # load annotations
        if split == "train":
            anno_path = os.path.join(self.data_dir, "nlq_train.json")
        elif split == "val":  # use val set for test
            anno_path = os.path.join(self.data_dir, "nlq_val.json")
        elif split == "test":  # TODO delete this line for final release
            anno_path = os.path.join(self.data_dir, "nlq_test_unannotated.json") # .json") # _unannotated.json")
        with open(anno_path) as f:
            anno_json = json.load(f)

        anno_pairs = []
        query_loop_count = 0
        for video_count, anno_video in enumerate(anno_json["videos"]):
            video_name = anno_video['video_uid'] # anno_clip["clip_uid"]
            for anno_clip in anno_video["clips"]:
                clip_times = [float(anno_clip["video_start_sec"]), float(anno_clip["video_end_sec"])]  #
                clip_duration = clip_times[1] - clip_times[0]
                clip_uid =  anno_clip["clip_uid"]
                # - enumerate annotations
                for anno in anno_clip["annotations"]:
                    anno_uid = anno['annotation_uid']
                    for query_idx, query in enumerate(anno["language_queries"]):
                        if split == 'test':
                            query_times = 0,0
                        else:
                            query_times = float(query["clip_start_sec"]), float(query["clip_end_sec"])
                        query_duration = (
                            query_times[1] - query_times[0]
                        )  # in terms of seconds

                        if split == "train":  # split to windows
                            if self.min_duration <= query_duration < self.window and (
                                not self.debug or video_count < 8
                            ):
                                # find the new start and end time in the window
                                first_window_start = np.ceil((query_times[0] - self.window) / stride) * stride
                                
                                last_window_start = query_times[1] // stride * stride
                                # use negative windows for none-action clips
                                if config.DATASET.NEGWINDOW:
                                    first_window_start += -stride * 2
                                    last_window_start += stride * 2
                                first_window_start = max(0, int(first_window_start))
                                last_window_start = min(int(clip_duration), int(last_window_start))
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
                                        "clip_se": clip_times, # tart": clip_times[0],
                                        "description": query["query"],
                                        "window": [w_start, w_start + self.window],
                                        "clip_duration": clip_duration,
                                        "times": [
                                            query_times[0] - w_start,
                                            query_times[1] - w_start,
                                        ],
                                        'query_uid': anno_uid+'_'+query["query"],
                                    }
                                    if w_start < clip_duration:
                                        anno_pairs.append(new_anno)

                        else:  # for val/test set, we need to process all windows
                            if split == 'val':
                                if self.min_duration > query_duration or query_duration > self.window or (
                                    self.debug and video_count > 1 # only for debug
                                ):
                                    break
                            else: # test set does not remove any query
                                query_loop_count += 1
                                new_anno = None
                                if int(clip_duration) - self.window + stride <= stride:
                                    print('warning:', int(clip_duration), self.window, stride)
                                for w_start in range(
                                    0, int(clip_duration) - self.window + stride, stride
                                ):
                                    new_anno = {
                                        "video": video_name,
                                        "clip": clip_uid, # used in evaluation server
                                        "clip_se": clip_times,
                                        "description": query["query"],
                                        "window": [w_start, w_start + self.window],
                                        "clip_duration": clip_duration,
                                        "times": [query_times[0], query_times[1]],
                                        "query_uid": anno_uid+'_'+str(query_idx),
                                        "query_idx": query_idx,
                                    }
                                    if (
                                        self.temp is None
                                        or anno_df["query"].values[i]
                                        in self.query_template[self.temp]
                                    ):
                                        anno_pairs.append(new_anno)
                                if new_anno is None:
                                    print('Warning!')

        print(
            " -- collected {} samples for dataset {}".format(
                len(anno_pairs), (split, self.temp)
            )
        )
        all_query_set = set()
        for anno in anno_pairs:
            all_query_set.add(anno['query_uid'])
        print(" -- number of queries: {}".format(len(all_query_set)))
        print(" -- query loop count: {}".format(query_loop_count))
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
        clip_se = self.annotations[index]["clip_se"]
        window_se = self.annotations[index]["window"]
        query_idx = self.annotations[index]["query_idx"]
        clip_uid = self.annotations[index]["clip"]

        window_start = window_se[0]
        window_end = window_se[1]
        duration = window_end - window_start

        # -- we use bert embedding
        if sentence not in self.cache_bert_feature:
            bert_feature = self.cache_text_feature(sentence, rnd=(config.DATASET.TXT_FEATURE=='rnd'))
            self.cache_bert_feature[sentence] = bert_feature
        word_vectors = self.cache_bert_feature[sentence]

        visual_input, visual_mask = self.get_video_features(
            video_id, video_duration, [window_start, window_end], clip_se
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
            "query_idx": query_idx,
            "clip": clip_uid
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def cache_text_feature(self, sentence, rnd=False):
        sentence = sentence.lower().strip("? \n.") + "?"
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {key: val.cuda() for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        output = outputs[0].data.detach()
        cls_feature = output[0] 
        if rnd:
            cls_feature = torch.randn_like(cls_feature)
        # print('given input {}, get {}'.format(inputs,output.shape))
        return cls_feature  # dim:768

    def get_video_features(self, vid, duration, window_se=None, clip_se=None):
        if "slowfast" == config.DATASET.VIS_INPUT_TYPE:
            fps = 30.0/16 # feature per second. assume
            feature = torch.load(self.data_dir + "/sf/{}.pt".format(vid))

            # features = np.load(self.data_dir + '/2d/{}.npy'.format(vid))
            features = torch.tensor(feature).float()
            # duration = features.shape[0]*fps
            # fps = 1.0 * features.shape[0] / duration

        elif "slowfast_old" == config.DATASET.VIS_INPUT_TYPE:
            fps = 30.0/16 # feature per second. assume
            feature = torch.load(self.data_dir + "/sf_old/{}.pt".format(vid))

            # features = np.load(self.data_dir + '/2d/{}.npy'.format(vid))
            features = torch.tensor(feature).float()
            # duration = features.shape[0]*fps
            # fps = 1.0 * features.shape[0] / duration

        elif "rnd" == config.DATASET.VIS_INPUT_TYPE:
            fps = 30.0/16 # feature per second. assume
            # if file does not exist, we will create one using randn_like
            if os.path.exists(self.data_dir + "/rnd/{}.pt".format(vid)):
                feature = torch.load(self.data_dir + "/rnd/{}.pt".format(vid))
            else:
                feature = torch.load(self.data_dir + "/sf/{}.pt".format(vid))
                feature = torch.randn_like(feature)
                torch.save(feature, self.data_dir + "/rnd/{}.pt".format(vid))

            features = torch.tensor(feature).float()
        else:
            raise NotImplementedError()

        if clip_se is not None:
            feat_start = int(clip_se[0] * fps)
            feat_end = int(clip_se[1] * fps)
            features = features[feat_start:feat_end, :]
            assert (features.shape[0] > 0), "meet clip feat se {} for video {}: {}, {} in duration: {}".format(feat_start, feat_end, vid, features.shape, duration, clip_se)

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

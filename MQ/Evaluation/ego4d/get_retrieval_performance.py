
import numpy as np
import os
import json
class Moment_Retrieval(object):
    GROUND_TRUTH_FIELDS = ['database']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset='test', verbose=False,
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        # Import ground truth and predictions.
        self.ground_truth = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):

        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)


        ground_truth = {}
        for videoid, v in data.items():

            if not v['subset'] in self.subset:
                continue

            annotations = {}
            for ann in v['annotations']:

                if ann['label'] not in annotations.keys():
                    annotations[ann['label']] = []
                annotations[ann['label']].append([ann['start_time'], ann['end_time']])

            ground_truth[v['clip_id']] = annotations

        return ground_truth

    def _import_prediction(self, prediction_filename):

        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')


        prediction = {}
        for videoid, v in data['results'].items():

            pred_props = {}
            for prop in v:
                if prop['label'] not in pred_props.keys():
                    pred_props[prop['label']] = []
                pred_props[prop['label']].append([prop['segment'][0], prop['segment'][1], prop['score']])

            prediction[videoid] = pred_props

        return prediction

    def evaluate(self):
        tious = [0.3, 0.5, 0.7]
        recalls = [1, 2, 3, 4, 5]

        # eval_result = [[[ [] for _ in range(len(self.ground_truth))] for _ in recalls] for _ in tious]
        eval_result = [[ [] for _ in recalls] for _ in tious]

        # v_cnt = 0
        for key_v, value_v in self.ground_truth.items():
            gt_v = value_v
            pred_v = self.prediction[key_v]

            for key_label, value_c in gt_v.items():
                gt_v_c = value_c
                num_gt_v_c = len(gt_v_c)
                if key_label in pred_v.keys():
                    pred_v_c = pred_v[key_label]
                    overlap = iou(pred_v_c, gt_v_c)

                    for i, t in enumerate(tious):
                        for j, r in enumerate(recalls):

                            is_retrieved = [(overlap > t)[:r*num_gt_v_c][:,i].any() for i in range(num_gt_v_c)]
                            eval_result[i][j].extend(is_retrieved)
                else:
                    for i, t in enumerate(tious):
                        for j, r in enumerate(recalls):
                            eval_result[i][j].extend([False] * len(gt_v_c))


        eval_result = np.array(eval_result).mean(axis=-1)

        for i, t in enumerate(tious):
            for j, r in enumerate(recalls):
                recall = eval_result[i, j]
                print(f'Rank {r}x @ tIoU {t} is {recall}')

        return eval_result


def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def evaluation_retrieval(opt):

    ego4d_MR = Moment_Retrieval(ground_truth_filename = opt["clip_anno"],
                   prediction_filename = os.path.join(opt["output_path"], opt["retrieval_result_file"]),
                   subset=opt['infer_datasplit'], tiou_thresholds=opt['tIoU_thr'],
                   verbose=True, check_status=False)

    eval_result = ego4d_MR.evaluate()

    print(eval_result)

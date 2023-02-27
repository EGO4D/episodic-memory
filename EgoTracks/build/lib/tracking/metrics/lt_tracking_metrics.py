import math
from typing import Iterable, List

import numpy as np
from tqdm import tqdm
from tracking.metrics.miou import compute_overlaps

# Copied from https://github.com/votchallenge/toolkit/blob/master/vot/analysis/tpr.py
def determine_thresholds(scores: Iterable[float], resolution: int) -> List[float]:
    scores = [
        score for score in scores if not math.isnan(score)
    ]  # and not score is None]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(
            np.linspace(delta, len(scores) - delta, num=resolution - 2)
        ).astype(np.int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores

    thresholds.insert(0, math.inf)
    thresholds.insert(len(thresholds), -math.inf)

    return thresholds


def compute_tpr_curves(confidences, overlaps, thresholds, n_visible):
    precision = len(thresholds) * [float(0)]
    recall = len(thresholds) * [float(0)]
    for i, threshold in enumerate(thresholds):

        subset = confidences >= threshold

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[subset])
            recall[i] = np.sum(overlaps[subset]) / n_visible

    return precision, recall


def compute_precision_and_recall(
    all_pred_scores: List[List], all_pred_bboxes: List[List], all_gt_bboxes: List[List]
):
    """
    Compute the precision and recall for the entire dataset. Per score/bbox for each frame of each video.
    If there is no gt bbox for that frame, then it is set to None.

    all_pred_scores: confidence score for all tracking trajectories in the dataset.
    all_pred_bboxes: predicted bbox for all tracking trajectories in the dataset.
    all_gt_bboxes: gt bbox for all tracking trajectories in the dataset.
    """
    all_overlaps = [
        compute_overlaps(pred_bboxes, gt_bbboxes)
        for pred_bboxes, gt_bbboxes in zip(all_pred_bboxes, all_gt_bboxes)
    ]
    resolution = 100

    thresholds = determine_thresholds(
        [s for conf in all_pred_scores for s in conf], resolution
    )

    precision = len(thresholds) * [float(0)]
    recall = len(thresholds) * [float(0)]

    for i in tqdm(range(len(all_gt_bboxes)), total=len(all_gt_bboxes)):
        confidences = np.array(all_pred_scores[i])
        overlaps = np.array(all_overlaps[i])
        gt_bboxes = all_gt_bboxes[i]
        n_visible = len([b for b in gt_bboxes if b is not None])

        pr, re = compute_tpr_curves(confidences, overlaps, thresholds, n_visible)
        for j in range(len(thresholds)):
            precision[j] += pr[j]
            recall[j] += re[j]

    precision = [pr / len(all_gt_bboxes) for pr in precision]
    recall = [re / len(all_gt_bboxes) for re in recall]

    return precision, recall


def compute_f_score(precision, recall):
    f_score = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    n = np.argmax(f_score)
    return (f_score[n], precision[n], recall[n])

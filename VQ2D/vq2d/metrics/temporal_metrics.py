from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..structures import ResponseTrack
from .utils import PRINT_FORMAT
from .utils import segment_iou, interpolated_prec_rec


# These are modified versions of the ActivityNet evaluation toolkit
# https://github.com/activitynet/ActivityNet


class TemporalDetection(object):

    metric_uuid: str = "Temporal AP"
    tiou_thresholds: np.ndarray = np.array([0.25, 0.5, 0.75, 0.95])
    tious_to_report: List[float] = [0.25]

    def __init__(
        self,
        ground_truth: List[ResponseTrack],
        prediction: List[List[ResponseTrack]],
        ignore_iou_averaging: bool = False,
    ):
        self.ap = None
        self.ground_truth = self._import_ground_truth(ground_truth)
        self.prediction = self._import_prediction(prediction)
        self.ignore_iou_averaging = ignore_iou_averaging

    def _import_ground_truth(self, ground_truth: List[ResponseTrack]) -> pd.DataFrame:
        """Converts input ground-truth to appropriate format."""
        video_lst, t_start_lst, t_end_lst = [], [], []
        for i, gt in enumerate(ground_truth):
            video_lst.append(i)
            t_start_lst.append(gt.temporal_extent[0])
            t_end_lst.append(gt.temporal_extent[1])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
            }
        )
        return ground_truth

    def _import_prediction(self, prediction: List[List[ResponseTrack]]) -> pd.DataFrame:
        """Converts input predictions to appropriate format."""
        video_lst, t_start_lst, t_end_lst, score_lst = [], [], [], []
        for i, preds in enumerate(prediction):
            # Iterate over each prediction
            for pred in preds:
                assert pred.has_score()
                video_lst.append(i)
                t_start_lst.append(pred.temporal_extent[0])
                t_end_lst.append(pred.temporal_extent[1])
                score_lst.append(pred.score)
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "score": score_lst,
            }
        )
        return prediction

    def evaluate(self) -> None:
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = compute_average_precision_detection(
            self.ground_truth, self.prediction, self.tiou_thresholds
        )

        self.average_ap = self.ap.mean().item()

    def get_metrics(self) -> Dict[str, float]:
        self.evaluate()
        metrics = OrderedDict()
        avg_suffix = (
            f"@ IoU={self.tiou_thresholds[0]:.2f}:{self.tiou_thresholds[-1]:.2f}"
        )
        if not self.ignore_iou_averaging:
            metrics[PRINT_FORMAT.format(self.metric_uuid, avg_suffix)] = self.average_ap
        for tiou_idx, tiou_thr in enumerate(self.tiou_thresholds):
            if tiou_thr not in self.tious_to_report:
                continue
            metrics[
                PRINT_FORMAT.format(self.metric_uuid, f"@ IoU={tiou_thr:.2f}")
            ] = self.ap[tiou_idx].item()
        return metrics


def compute_average_precision_detection(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: Sequence[float] = np.linspace(0.5, 0.95, 10),
) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap

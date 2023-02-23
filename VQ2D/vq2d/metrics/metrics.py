from collections import OrderedDict
from typing import List, Dict, Tuple

import numpy as np

from .spatio_temporal_metrics import SpatioTemporalDetection
from .success_metrics import SuccessMetrics
from .temporal_metrics import TemporalDetection
from .tracking_metrics import TrackingMetrics
from .utils import BBox, ResponseTrack


METRIC_FNS = [
    lambda gt, pred: TemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: SpatioTemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: TrackingMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
    lambda gt, pred: SuccessMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
]


def compute_visual_query_metrics(
    predicted_response_track: List[List[ResponseTrack]],
    ground_truth_response_track: List[ResponseTrack],
    visual_crop_boxes: List[BBox],
    accessed_frames_in_clip: List[int] = None,
    total_frames_in_clip: List[int] = None,
    area_ranges: Dict[str, List[int]] = {
        "all": [0 ** 2, 1e5 ** 2],
        "medium": [32 ** 2, 96 ** 2],
        "large": [96 ** 2, 1e5 ** 2],
    },
    vc_rt_pairings: Dict[str, Tuple[str, str]] = {
        "all": ("all", "all"),
    },
) -> Dict[str, float]:
    """
    Compute model performance on the visual query task. Includes the following metrics:
        * Temporal AP
        * SpatioTemporal AP
        * Success
        * Tracking % recovery
        * Search efficiency
    """

    # Calculate visual-crop areas
    vc_areas = np.array(
        [
            abs(vc_bbox.x2 - vc_bbox.x1) * abs(vc_bbox.y2 - vc_bbox.y1)
            for vc_bbox in visual_crop_boxes
        ]
    )
    # Calculate response-track max areas
    rt_areas = []
    for rt in ground_truth_response_track:
        area = (
            np.array(
                [
                    abs(rt_bbox.x2 - rt_bbox.x1) * abs(rt_bbox.y2 - rt_bbox.y1)
                    for rt_bbox in rt.bboxes
                ]
            )
            .max()
            .item()
        )
        rt_areas.append(area)
    rt_areas = np.array(rt_areas)

    # Calculate metrics for each vc_rt_pairing
    pair_metrics = OrderedDict()
    for pair_name, (vc_cat, rt_cat) in vc_rt_pairings.items():
        vc_range = area_ranges[vc_cat]
        rt_range = area_ranges[rt_cat]
        # Get data points satifying the pairing criterion
        mask = (
            (vc_areas >= vc_range[0])
            & (vc_areas < vc_range[1])
            & (rt_areas >= rt_range[0])
            & (rt_areas < rt_range[1])
        )
        # Ignore pairing if there are not valid data points
        if np.count_nonzero(mask) == 0:
            continue
        # Calculate metrics
        pred_rt = [predicted_response_track[i] for i, cond in enumerate(mask) if cond]
        gt_rt = [ground_truth_response_track[i] for i, cond in enumerate(mask) if cond]
        if accessed_frames_in_clip is not None:
            acc_frames = [
                accessed_frames_in_clip[i] for i, cond in enumerate(mask) if cond
            ]
            tot_frames = [
                total_frames_in_clip[i] for i, cond in enumerate(mask) if cond
            ]
        metrics = OrderedDict()
        for metric_fn in METRIC_FNS:
            metrics.update(metric_fn(gt_rt, pred_rt))
        if accessed_frames_in_clip is not None and len(acc_frames) > 0:
            metrics["Search efficiency (%)"] = (
                1 - np.array(acc_frames).astype(np.float32) / np.array(tot_frames)
            ).mean() * 100.0
        pair_metrics[pair_name] = metrics

    return pair_metrics

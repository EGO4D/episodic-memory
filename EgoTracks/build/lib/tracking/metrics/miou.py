import numpy as np
from tracking.utils.bbox_helper import IoU, xywh_2_rect


def compute_overlaps(y_pred, y_gt):
    overlaps = []
    for p, g in zip(y_pred, y_gt):
        if p is None or g is None:
            overlap = 0
        else:
            p = xywh_2_rect(p)
            g = xywh_2_rect(g)
            overlap = IoU(p, g)
        overlaps.append(overlap)

    return overlaps


def mIoU(y_pred, y_gt):
    overlaps = compute_overlaps(y_pred, y_gt)

    return np.mean(overlaps)

#!/usr/bin/env python3
"""
    Tests for the eval metrics.

    To run this test:
    >>> buck test @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tests:eval_metrics_test -- --print-passing-details

"""
import unittest

from tracking.metrics.lt_tracking_metrics import (
    compute_f_score,
    compute_precision_and_recall,
)


class TestEvalMetrics(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEvalMetrics, self).__init__(*args, **kwargs)

    def test_tracking_precision_and_recall(self):
        all_pred_scores = [[0.9, 0.8, 0.95], [0.87, 0.4]]
        all_pred_bboxes = [
            [[50, 48, 100, 100], [47, 47, 100, 100], [49, 49, 100, 100]],
            [[60, 50, 100, 100], [60, 60, 100, 100]],
        ]
        all_gt_bboxes = [
            [[50, 50, 100, 100], [50, 50, 100, 100], [50, 50, 100, 100]],
            [[50, 50, 100, 100], None],
        ]
        precision, recall = compute_precision_and_recall(
            all_pred_scores, all_pred_bboxes, all_gt_bboxes
        )
        self.assertEqual(len(precision), len(recall))

        f1 = compute_f_score(precision, recall)
        print(f1)

from .eval_ego4d_lt_tracking import (
    calculate_ego4d_lt_tracking_metrics,
    eval_ego4d_lt_tracking,
)
from .eval_ego4d_vq_tracking import (
    calculate_ego4d_vq_tracking_metrics,
    eval_ego4d_vq_tracking,
)
from .eval_got10k import calculate_got10k_metrics, eval_got10k


EVAL_FUNCTIONS = {
    "EGO4DVQTracking": eval_ego4d_vq_tracking,
    "GOT10K": eval_got10k,
    "EGO4DLTTracking": eval_ego4d_lt_tracking,
}
CALCULATE_METRICS_FUNCTIONS = {
    "EGO4DVQTracking": calculate_ego4d_vq_tracking_metrics,
    "EGO4DLTTracking": calculate_ego4d_lt_tracking_metrics,
    "GOT10K": calculate_got10k_metrics,
}


def build_eval_function(dataset_name):
    return EVAL_FUNCTIONS[dataset_name]


def build_calculate_metrics_function(dataset_name):
    return CALCULATE_METRICS_FUNCTIONS[dataset_name]

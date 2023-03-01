from .eval_ego4d_lt_tracking import (
    eval_ego4d_lt_tracking,
)

EVAL_FUNCTIONS = {
    "EGO4DLTTracking": eval_ego4d_lt_tracking,
}
CALCULATE_METRICS_FUNCTIONS = {
    "EGO4DLTTracking": None,
}


def build_eval_function(dataset_name):
    return EVAL_FUNCTIONS[dataset_name]


def build_calculate_metrics_function(dataset_name):
    return CALCULATE_METRICS_FUNCTIONS[dataset_name]

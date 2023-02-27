# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import logging

import detectron2.utils.comm as comm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tracking.models.stark_tracker.stark_tracker import STARKTracker
from .eval_datasets.build import (
    build_calculate_metrics_function,
    build_eval_function,
)
from tracking.utils.defaults import setup


parser = argparse.ArgumentParser(description="Tracking")
MODELS = {
    "STARK": STARKTracker,
}


def modify_STARK_cfg_by_dataset(cfg, dataset_name):
    cfg.defrost()
    if dataset_name == "EGO4DVQTracking":
        cfg.TEST.IS_SEARCH_LOCAL = cfg.EVAL.EGO4DVQ.IS_SEARCH_LOCAL

    cfg.TEST.UPDATE_INTERVALS.UPDATE_INTERVALS = getattr(
        cfg.TEST.UPDATE_INTERVALS, dataset_name
    )
    cfg.freeze()

    return cfg


def eval_main(args, cfg=None):
    # The optional cfg is used for training code to overwrite
    # which model checkpoint to read from
    if cfg is None:
        cfg = setup(args)
    local_rank = comm.get_local_rank()
    logging.info("Use GPU: {} for evaluating".format(local_rank))

    cudnn.benchmark = False
    cudnn.deterministic = True

    # create model

    for dataset_name in cfg.EVAL.EVAL_DATASETS:
        logging.info(f"Creating model {cfg.MODEL_TYPE} for dataset {dataset_name}")
        # In STARK, we need to set the update interval for different datasets
        if cfg.MODEL_TYPE == "STARK":
            cfg = modify_STARK_cfg_by_dataset(cfg, dataset_name)
        logging.info(cfg)
        if cfg.MODEL_TYPE in MODELS:
            model = MODELS[cfg.MODEL_TYPE](
                cfg, device=torch.device(f"cuda:{local_rank}")
            )
        else:
            raise NotImplementedError(f"Model type {cfg.MODEL_TYPE} is not supported!")
        eval_func = build_eval_function(dataset_name)
        eval_func(model, cfg)


def calculate_metrics(args):
    cfg = setup(args)
    result = {}
    for dataset_name in cfg.EVAL.EVAL_DATASETS:
        calculate_metrics_func = build_calculate_metrics_function(dataset_name)
        result[dataset_name] = calculate_metrics_func(cfg)

    return result


if __name__ == "__main__":
    eval_main()

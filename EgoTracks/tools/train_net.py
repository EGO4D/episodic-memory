"""
Train_net for Tracking

Run this command to test locally:
>>> buck run @mode/inplace @mode/dev-nosan //vision/fair_accel/pixar_env/pixar_environment/tracking/tools:train_net -- --print-passing-details
"""

from collections import defaultdict
import csv
import json
import logging
import os
from datetime import datetime

import detectron2.utils.comm as comm
import torch
import yaml
from torch import distributed as dist, multiprocessing as mp
from torch.backends import cudnn
from tracking.tools.eval_net import eval_main
from tracking.tools.trainers.starkst_trainer import STARKSTrainer, STARKSTTrainer
from tracking.utils.defaults import default_argument_parser, setup
from tracking.utils.env import pathmgr
from tracking.utils.types import Params
from tracking.utils.multiprocessing import launch_job
import pickle as pkl
import io


# # TODO: delete once finish debug
# def train_main(params, rank=0, gpu=0, group=None):
#     OUTPUT_DIR = params.result_dir
#     cudnn.benchmark = True
#     print("Start training fisrt stage - STARKS ...")
#     params.STARK.model_type = "stark_st1"
#     params.STARK.TRAIN.TRAIN_CLS = False
#     params.STARK.DATA.TRAIN.DATASETS_NAME = ["LASOT"]
#     params.STARK.DATA.TRAIN.DATASETS_RATIO = [1]
#     params.STARK.DATA.DATA_FRACTION = None  # Faster test

#     params.STARK.TRAIN.EPOCH = 2
#     params.STARK.TRAIN.BATCH_SIZE = 4

#     # decrease iterations to test faster
#     params.STARK.DATA.TRAIN.SAMPLE_PER_EPOCH = 40
#     params.STARK.DATA.VAL.SAMPLE_PER_EPOCH = 20
#     params.STARK.TRAIN.VAL_EPOCH_INTERVAL = 5

#     params.STARK.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "STARKS")

#     starks_trainer = STARKSTrainer(params.STARK)
#     starks_trainer.train()

#     comm.synchronize()
#     print("Start training second stage - STARKST ...")

#     params.STARK.model_type = "stark_st2"
#     params.STARK.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "STARKST")
#     params.STARK.PREV_CHECKPOINT_DIR = os.path.join(
#         OUTPUT_DIR, "STARKS", "checkpoints"
#     )
#     params.STARK.TRAIN.TRAIN_CLS = True
#     starkst_trainer = STARKSTTrainer(params.STARK)
#     print(f"{comm.get_rank()} Trainer {STARKSTTrainer} {starkst_trainer}")
#     starkst_trainer.train(load_previous_ckpt=True)
#     # train_main(params, rank=0, local_rank=-1, group=None)

#     comm.synchronize()
#     # eval_result
#     checkpoint_dir = os.path.join(
#         OUTPUT_DIR,
#         "STARKST",
#         "checkpoints",
#     )
#     checkpoint_list = pathmgr.ls(checkpoint_dir)
#     checkpoint_list = sorted(checkpoint_list)
#     checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
#     params.model_path = checkpoint_path
#     params.result_dir = os.path.join(
#         OUTPUT_DIR,
#         "eval",
#         f"{params.track_mode}_{'5FPS' if params.is_read_5FPS_clip else '30FPS'}",
#     )
#     local_rank = comm.get_local_rank()
#     global_rank = comm.get_rank()
#     return eval_main(params, global_rank, local_rank)


def train_main(args):
    logging.info(args)
    cfg = setup(args)
    output_dir = cfg.OUTPUT_DIR
    pathmgr.mkdirs(output_dir)
    if comm.is_main_process():
        with pathmgr.open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg.dump(), f)

    if cfg.MODEL_TYPE == "STARK":
        stage_1_cfg = cfg.clone()
        stage_1_cfg.defrost()
        # STARK requires two-stage training:
        # First stage train localization while
        # the second stage train classification
        if comm.is_main_process():
            logging.error("Start training fisrt stage - STARKS ...")
        cudnn.benchmark = cfg.CUDNN_BENCHMARK
        stage_1_cfg.TRAIN = cfg.TRAIN_STAGE_1.clone()
        stage_1_cfg.OUTPUT_DIR = os.path.join(output_dir, "STARKS")
        logging.info(stage_1_cfg)
        stage_1_cfg.freeze()

        starks_trainer = STARKSTrainer(stage_1_cfg)
        if stage_1_cfg.MODEL.WEIGHTS:
            starks_trainer.load_checkpoint(checkpoint=stage_1_cfg.MODEL.WEIGHTS)
        if args.resume:
            starks_trainer.load_checkpoint()
        starks_trainer.train()

        comm.synchronize()
        if comm.is_main_process():
            logging.error("Start training second stage - STARKST ...")

        stage_2_cfg = cfg.clone()
        stage_2_cfg.defrost()
        stage_2_cfg.TRAIN = cfg.TRAIN_STAGE_2.clone()
        stage_2_cfg.OUTPUT_DIR = os.path.join(output_dir, "STARKST")
        stage_2_cfg.freeze()
        prev_checkpoint_dir = os.path.join(output_dir, "STARKS", "checkpoints")

        # Load the checkpoint from Stage1 training
        checkpoint_list = pathmgr.ls(prev_checkpoint_dir)
        checkpoint_list = sorted(checkpoint_list)
        checkpoint_path = os.path.join(prev_checkpoint_dir, checkpoint_list[-1])

        starkst_trainer = STARKSTTrainer(stage_2_cfg)
        starkst_trainer.load_checkpoint(checkpoint=checkpoint_path)
        starkst_trainer.train()

        comm.synchronize()
    else:
        raise NotImplementedError(
            f"Training model type {cfg.MODEL_TYPE} is not supported!"
        )

    # eval_result
    eval_cfg = cfg.clone()
    eval_cfg.defrost()
    checkpoint_dir = os.path.join(
        output_dir,
        "STARKST",
        "checkpoints",
    )
    checkpoint_list = pathmgr.ls(checkpoint_dir)
    checkpoint_list = sorted(checkpoint_list)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
    eval_cfg.MODEL.WEIGHTS = checkpoint_path
    eval_cfg.freeze()

    return eval_main(args, cfg=eval_cfg)


def run_train(
    local_rank,
    main_func,
    params,
    num_machines,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
):
    world_size = num_machines * num_gpus_per_machine
    rank = machine_rank * num_gpus_per_machine + local_rank
    dist.init_process_group(
        backend="NCCL",
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )
    assert comm._LOCAL_PROCESS_GROUP is None
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    torch.cuda.set_device(local_rank)
    main_func(params, rank=rank, gpu=local_rank)


def result2submission(args):
    cfg = setup(args)
    result_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "eval",
        "EGO4DLTTracking",
        f"{cfg.EVAL.EGO4DLT.TRACK_MODE}",
        f"{cfg.MODEL_TYPE}",
    )
    intermediate_dir = os.path.join(result_dir, "intermediate_result")

    result = {}

    files = pathmgr.ls(intermediate_dir)
    for f in files:
        path = os.path.join(intermediate_dir, f)
        logging.info(path)
        shard_result = pkl.load(pathmgr.open(path, "rb"))
        result.update(shard_result)

    submission = defaultdict(dict)
    for k, v in result.items():
        data = []
        for bbox_dict in v["pred_bboxes"]:
            submission[k][int(bbox_dict["frame_number"])] = bbox_dict["bbox"] + [
                bbox_dict["score"]
            ]

    with open(os.path.join(result_dir, f"submission.json"), "w") as f:
        json.dump(submission, f, indent=4)


def main():
    args = default_argument_parser().parse_args()

    args.config_file = "configs/STARK/stark_st_base.yaml"
    if args.eval_only:
        launch_job(
            eval_main,
            args.num_gpus,
            num_machines=args.num_machines,
            init_method=args.dist_url,
            machine_rank=args.machine_rank,
            args=(args,),
        )
        result2submission(args)
    else:
        launch_job(
            train_main,
            args.num_gpus,
            num_machines=args.num_machines,
            init_method=args.dist_url,
            machine_rank=args.machine_rank,
            args=(args,),
        )


if __name__ == "__main__":
    main()

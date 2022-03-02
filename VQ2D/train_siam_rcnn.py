#!/usr/bin/env python

import glob
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2_extensions.config import get_cfg
from torch.nn.parallel import DistributedDataParallel
from vq2d.baselines import VisualQueryDatasetMapper, register_visual_query_datasets


logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluators.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(
            cfg, dataset_name, mapper=VisualQueryDatasetMapper(cfg, is_train=False)
        )
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    return results


def set_model_to_train(model):
    """
    Freezes backbone, proposal_generator. Sets roi_heads to train.
    """
    # Freeze backbone
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Freeze proposal_generator
    model.proposal_generator.eval()
    for p in model.proposal_generator.parameters():
        p.requires_grad = False
    # Set roi_heads to train
    model.roi_heads.train()


def do_train(cfg, model, resume=False):
    distributed = comm.get_world_size() > 1
    if distributed:
        set_model_to_train(model.module)
    else:
        set_model_to_train(model)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # This script does not support accurate timing and precise BN here.
    # They are not trivial to implement in a small training loop.
    data_loader = build_detection_train_loader(
        cfg,
        mapper=VisualQueryDatasetMapper(cfg, is_train=True),
    )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
                comm.synchronize()
                # Required again since do_test sets full model to train() at the end
                if distributed:
                    set_model_to_train(model.module)
                else:
                    set_model_to_train(model)
                # Log results to storage
                if comm.is_main_process():
                    results_formed = {}
                    for dset, results_i in results.items():
                        for rtype, results_ij in results_i.items():
                            for metric, result_metric in results_ij.items():
                                results_formed[
                                    f"{dset}/{rtype}/{metric}"
                                ] = result_metric
                    storage.put_scalars(**results_formed)

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_siam_model(cfg):
    model = build_model(cfg)
    # Initialize backbone, proposal_generators
    pretrained_model = model_zoo.get(cfg.MODEL.SIAMESE_PRETRAINED_CONFIG, trained=True)
    model.backbone.load_state_dict(pretrained_model.backbone.state_dict())
    model.proposal_generator.load_state_dict(
        pretrained_model.proposal_generator.state_dict()
    )
    return model


def register_all_datasets(cfg):
    # Register VQ datasets
    # The dataset names are "<NAME>_<SPLIT>" for SPLIT in "train", "val", "test"
    splits_root = cfg.INPUT.VQ_DATA_SPLITS_ROOT
    images_root = cfg.INPUT.VQ_IMAGES_ROOT
    register_visual_query_datasets(splits_root, images_root, "visual_query")
    register_visual_query_datasets(
        splits_root,
        images_root,
        "visual_query_clean",
        bbox_aspect_scale=0.5,
        bbox_area_scale=0.25,
    )
    register_visual_query_datasets(
        splits_root,
        images_root,
        "visual_query_clean_plus",
        bbox_aspect_scale=0.75,
        bbox_area_scale=0.50,
    )
    register_visual_query_datasets(
        splits_root,
        images_root,
        "visual_query_clean_aug",
        bbox_aspect_scale=0.5,
        bbox_area_scale=0.25,
        perform_response_augmentation=True,
    )


def main(args):
    cfg = setup(args)

    register_all_datasets(cfg)

    model = build_siam_model(cfg)

    if not args.eval_only:
        # Slurm resumption logic
        ## If a checkpoint exists, load the most recent one
        model_paths = glob.glob(os.path.join(cfg.OUTPUT_DIR, "model_*.pth"))
        if len(model_paths) > 0:
            get_ckpt_id = lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
            model_paths = sorted(model_paths, key=get_ckpt_id, reverse=True)
            cfg.defrost()
            cfg.MODEL.WEIGHTS = model_paths[0]
            cfg.freeze()
            args.resume = True
            print(f"======> Resuming training from {model_paths[0]}")

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    set_model_to_train(model)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,  # , find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

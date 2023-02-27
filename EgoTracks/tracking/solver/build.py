import logging

import detectron2.utils.comm as comm
import torch


def build_optimizer_scheduler(cfg, model):
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    return optimizer, scheduler


def build_optimizer(cfg, model):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        logging.info(
            "Only training classification head. Learnable parameters are shown below."
        )
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "cls" in n and p.requires_grad
                ]
            }
        ]

        for n, p in model.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                logging.info(n)
    else:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if comm.is_main_process():
            logging.info("Learnable parameters are shown below.")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported Optimizer")

    return optimizer


def build_scheduler(cfg, optimizer):
    if cfg.TRAIN.SCHEDULER.TYPE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg.TRAIN.LR_DROP_EPOCH
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA,
        )
    else:
        raise ValueError("Unsupported scheduler")
    return lr_scheduler


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print(
            "Only training classification head. Learnable parameters are shown below."
        )
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "cls" in n and p.requires_grad
                ]
            }
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if comm.is_main_process():
            logging.info("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    logging.info(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg.TRAIN.LR_DROP_EPOCH
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA,
        )
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler

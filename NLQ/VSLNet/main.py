"""Main script to train/test models for Ego4D NLQ dataset.
"""
import argparse
import os

import numpy as np
import options
import torch
import torch.nn as nn
from model.VSLNet import build_optimizer_and_scheduler, VSLNet
from tqdm import tqdm
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_test_loader, get_train_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)


def main(configs, parser):
    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset.get("n_chars", -1)
    configs.word_size = dataset.get("n_words", -1)

    # get train and test loader
    visual_features = load_video_features(
        os.path.join("data", "features", configs.task, configs.fv), configs.max_pos_len
    )
    # If video agnostic, randomize the video features.
    if configs.video_agnostic:
        visual_features = {
            key: np.random.rand(*val.shape) for key, val in visual_features.items()
        }
    train_loader = get_train_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    train_eval_loader = get_test_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    val_loader = (
        None
        if dataset["val_set"] is None
        else get_test_loader(dataset["val_set"], visual_features, configs)
    )
    test_loader = get_test_loader(
        dataset=dataset["test_set"], video_features=visual_features, configs=configs
    )
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)
    num_val_batches = 0 if val_loader is None else len(val_loader)
    num_test_batches = len(test_loader)

    # Device configuration
    cuda_str = "cuda" if configs.gpu_idx is None else "cuda:{}".format(configs.gpu_idx)
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")

    # create model dir
    home_dir = os.path.join(
        configs.model_dir,
        "_".join(
            [
                configs.model_name,
                configs.task,
                configs.fv,
                str(configs.max_pos_len),
                configs.predictor,
            ]
        ),
    )
    if configs.suffix is not None:
        home_dir = home_dir + "_" + configs.suffix
    model_dir = os.path.join(home_dir, "model")

    # train and test
    if configs.mode.lower() == "train":
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        eval_period = num_train_batches // 2
        save_json(
            vars(configs),
            os.path.join(model_dir, "configs.json"),
            sort_keys=True,
            save_pretty=True,
        )
        # build model
        model = VSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
        # start training
        best_metric = -1.0
        score_writer = open(
            os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        global_step = 0
        for epoch in range(configs.epochs):
            model.train()
            for data in tqdm(
                train_loader,
                total=num_train_batches,
                desc="Epoch %3d / %3d" % (epoch + 1, configs.epochs),
            ):
                global_step += 1
                (
                    _,
                    vfeats,
                    vfeat_lens,
                    word_ids,
                    char_ids,
                    s_labels,
                    e_labels,
                    h_labels,
                ) = data
                # prepare features
                vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                s_labels, e_labels, h_labels = (
                    s_labels.to(device),
                    e_labels.to(device),
                    h_labels.to(device),
                )
                if configs.predictor == "bert":
                    word_ids = {key: val.to(device) for key, val in word_ids.items()}
                    # generate mask
                    query_mask = (
                        (
                            torch.zeros_like(word_ids["input_ids"])
                            != word_ids["input_ids"]
                        )
                        .float()
                        .to(device)
                    )
                else:
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    # generate mask
                    query_mask = (
                        (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    )
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)
                # compute logits
                h_score, start_logits, end_logits = model(
                    word_ids, char_ids, vfeats, video_mask, query_mask
                )
                # compute loss
                highlight_loss = model.compute_highlight_loss(
                    h_score, h_labels, video_mask
                )
                loc_loss = model.compute_loss(
                    start_logits, end_logits, s_labels, e_labels
                )
                total_loss = loc_loss + configs.highlight_lambda * highlight_loss
                # compute and apply gradients
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), configs.clip_norm
                )  # clip gradient
                optimizer.step()
                scheduler.step()
                # evaluate
                if (
                    global_step % eval_period == 0
                    or global_step % num_train_batches == 0
                ):
                    model.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    result_save_path = os.path.join(
                        model_dir,
                        f"{configs.model_name}_{epoch}_{global_step}_preds.json",
                    )
                    # Evaluate on val, keep the top 3 checkpoints.
                    results, mIoU, score_str = eval_test(
                        model=model,
                        data_loader=val_loader,
                        device=device,
                        mode="val",
                        epoch=epoch + 1,
                        global_step=global_step,
                        gt_json_path=configs.eval_gt_json,
                        result_save_path=result_save_path,
                    )
                    print(score_str, flush=True)
                    score_writer.write(score_str)
                    score_writer.flush()
                    # Recall@1, 0.3 IoU overlap --> best metric.
                    if results[0][0] >= best_metric:
                        best_metric = results[0][0]
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                model_dir,
                                "{}_{}.t7".format(configs.model_name, global_step),
                            ),
                        )
                        # only keep the top-3 model checkpoints
                        filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                    model.train()
        score_writer.close()

    elif configs.mode.lower() == "test":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        # load previous configs
        pre_configs = load_json(os.path.join(model_dir, "configs.json"))
        parser.set_defaults(**pre_configs)
        configs = parser.parse_args()
        # build model
        model = VSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        # get last checkpoint file
        filename = get_last_checkpoint(model_dir, suffix="t7")
        model.load_state_dict(torch.load(filename))
        model.eval()
        result_save_path = filename.replace(".t7", "_test_result.json")
        results, mIoU, score_str = eval_test(
            model=model,
            data_loader=test_loader,
            device=device,
            mode="test",
            result_save_path=result_save_path,
        )
        print(score_str, flush=True)


if __name__ == "__main__":
    configs, parser = options.read_command_line()
    main(configs, parser)

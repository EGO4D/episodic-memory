import copy
import glob
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.evaluate_ego4d_nlq as ego4d_eval
from utils.data_util import index_to_time


def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix="t7", max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split("_")[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix="t7"):
    model_filenames = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split("_")[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def eval_test(
    model,
    data_loader,
    device,
    mode="test",
    result_save_path=None,
    gt_json_path=None,
    epoch=None,
    global_step=None,
):
    predictions = []
    with torch.no_grad():
        for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="evaluate {}".format(mode),
        ):
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)

            if isinstance(word_ids, dict):
                word_ids = {key: val.to(device) for key, val in word_ids.items()}
                # generate mask
                query_mask = (
                    (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                    .float()
                    .to(device)
                )
            else:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                # generate mask
                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)

            # generate mask
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute predicted results
            _, start_logits, end_logits = model(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )
            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()

            # Record output and use standard evalution script for NLQ.
            for record, starts, ends in zip(records, start_indices, end_indices):
                # Convert all indices to times.
                timewindow_predictions = []
                for start, end in zip(starts, ends):
                    start_time, end_time = index_to_time(
                        start, end, record["v_len"], record["duration"]
                    )
                    timewindow_predictions.append([float(start_time), float(end_time)])
                new_datum = {
                    "clip_uid": record["vid"],
                    "annotation_uid": record["annotation_uid"],
                    "query_idx": int(record["query_idx"]),
                    "predicted_times": copy.deepcopy(timewindow_predictions),
                }
                predictions.append(new_datum)

    # Save predictions if path is provided.
    if result_save_path:
        with open(result_save_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": predictions,
                }, file_id
            )

    # Evaluate if ground truth JSON file is provided.
    if gt_json_path:
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        title = f"Epoch {epoch}, Step {global_step}"
        display_results = ego4d_eval.display_results(
            results, mIoU, thresholds, topK, title=title
        )
    else:
        results = None
        mIoU = None
        display_results = None
    return results, mIoU, display_results

import gzip
import json
import os
import os.path as osp
import time
from queue import Empty as QueueEmpty

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pims
import skimage.io
import torch
import tqdm
from detectron2.utils.logger import setup_logger
from detectron2_extensions.config import get_cfg as get_detectron_cfg
from scipy.signal import find_peaks, medfilt
from torch import multiprocessing as mp
from vq2d.baselines import (
    create_similarity_network,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    perform_cached_retrieval,
    SiamPredictor,
)
from vq2d.structures import ResponseTrack
from vq2d.tracking import Tracker

setup_logger()

import hydra
from omegaconf import DictConfig, OmegaConf


def get_images_at_peak(all_bboxes, all_scores, all_imgs, peak_idx, topk=5):
    bboxes = all_bboxes[peak_idx]
    scores = all_scores[peak_idx]
    image = all_imgs[peak_idx]
    # Visualize the top K retrievals from peak image
    bbox_images = []
    for bbox in bboxes[:topk]:
        bbox_image = image[bbox.y1 : bbox.y2 + 1, bbox.x1 : bbox.x2 + 1]
        if bbox_image.shape[0] == 0 or bbox_image.shape[1] == 0:
            continue
        bbox_images.append(bbox_image)
    return bbox_images


class Task:
    def __init__(self, annot):
        super().__init__()
        self.annot = annot
        self.key = (annot["metadata"]["annotation_uid"], annot["metadata"]["query_set"])

    def run(self, predictor, similarity_net, tracker, cfg, device):

        data_cfg = cfg.data
        sig_cfg = cfg.signals

        start_time = time.time()
        clip_uid = self.annot["clip_uid"]
        annotation_uid = self.annot["metadata"]["annotation_uid"]
        query_set = self.annot["metadata"]["query_set"]
        annot_key = f"{annotation_uid}_{query_set}"
        # Load clip from file
        clip_path = os.path.join(
            data_cfg.data_root, get_clip_name_from_clip_uid(clip_uid)
        )
        if not os.path.exists(clip_path):
            print(f"Clip {clip_uid} does not exist")
            return None
        video_reader = pims.Video(clip_path)
        query_frame = self.annot["query_frame"]
        visual_crop = self.annot["visual_crop"]
        clip_read_time = time.time() - start_time
        start_time = time.time()
        # Retrieve nearest matches and their scores per image
        cached_bboxes, cached_scores, cache_exists = None, None, False
        if cfg.model.enable_cache:
            assert cfg.model.cache_root != ""
            cache_path = os.path.join(cfg.model.cache_root, f"{annot_key}.pt")
            if os.path.isfile(cache_path):
                cache = torch.load(cache_path)
                cached_bboxes = cache["ret_bboxes"]
                cached_scores = cache["ret_scores"]
                cache_exists = True
                assert len(cached_bboxes) == query_frame
                assert len(cached_scores) == query_frame
            else:
                print(f"Could not find cached detections: {cache_path}")

        if visual_crop["frame_number"] >= len(video_reader):
            print(
                "=====> VC {} is out of range in video {} with size {}".format(
                    visual_crop["frame_number"], clip_uid, len(video_reader)
                )
            )
            return None

        if cache_exists:
            ret_bboxes, ret_scores, ret_imgs, visual_crop_im = perform_cached_retrieval(
                video_reader,
                visual_crop,
                query_frame,
                predictor,
                cached_bboxes,
                cached_scores,
                recency_factor=cfg.model.recency_factor,
                subsampling_factor=cfg.model.subsampling_factor,
                visualize=cfg.logging.visualize,
            )
        else:
            ret_bboxes, ret_scores, ret_imgs, visual_crop_im = perform_retrieval(
                video_reader,
                visual_crop,
                query_frame,
                predictor,
                batch_size=data_cfg.rcnn_batch_size,
                recency_factor=cfg.model.recency_factor,
                subsampling_factor=cfg.model.subsampling_factor,
                visualize=cfg.logging.visualize,
            )
        detection_time_taken = time.time() - start_time
        start_time = time.time()
        # Generate a time signal of scores
        score_signal = []
        for scores in ret_scores:
            if len(scores) == 0:
                score_signal.append(0.0)
            else:
                score_signal.append(np.max(scores).item())
        # Smooth the signal using median filtering
        kernel_size = sig_cfg.smoothing_sigma
        if kernel_size % 2 == 0:
            kernel_size += 1
        score_signal_sm = medfilt(score_signal, kernel_size=kernel_size)
        # Identify the latest peak in the signal
        peaks, _ = find_peaks(
            score_signal_sm,
            distance=sig_cfg.distance,
            width=sig_cfg.width,
            prominence=sig_cfg.prominence,
        )
        peak_signal_time_taken = time.time() - start_time
        start_time = time.time()
        # Identify most recent peak with sufficient similarity
        recent_peak = None
        for peak in peaks[::-1]:
            if score_signal_sm[peak] >= sig_cfg.peak_similarity_thresh:
                recent_peak = peak
                # print(f"====> Signal peak score: {score_signal_sm[peak]}")
                break
        # Perform tracking to predict response track
        if recent_peak is not None:
            init_state = ret_bboxes[recent_peak][0]
            init_frame = video_reader[init_state.fno]
            pred_rt, pred_rt_vis = tracker(
                init_state, init_frame, video_reader, query_frame, similarity_net
            )
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]
        else:
            pred_rt = [ret_bboxes[-1][0]]
            pred_rt_vis = []
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]

        tracking_time_taken = time.time() - start_time
        print(
            "====> Data uid: {} | search window :{:>8d} frames | "
            "clip read time: {:>6.2f} mins | "
            "detection time: {:>6.2f} mins | "
            "peak signal time: {:>6.2f} mins | "
            "tracking time: {:>6.2f} mins".format(
                self.annot["clip_uid"],
                self.annot["query_frame"],
                clip_read_time / 60.0,
                detection_time_taken / 60.0,
                peak_signal_time_taken / 60.0,
                tracking_time_taken / 60.0,
            )
        )

        # Note: This visualization does not work for subsampled evaluation.
        if cfg.logging.visualize:
            print("Visualizing results...")
            ####################### Visualize the peaks ########################
            plt.figure(figsize=(6, 6))
            # Plot raw signals
            # plt.plot(score_signal, color="gray", label="Original signal")
            plt.plot(score_signal_sm, color="blue", label="Similarity scores")
            # Plot highest-scoring pred response track
            pred_rt_start, pred_rt_end = pred_rts[0].temporal_extent
            rt_signal = np.zeros((query_frame,))
            rt_signal[pred_rt_start : pred_rt_end + 1] = 1
            plt.plot(rt_signal, color="red", label="Pred response track")
            # Plot ground-truth response track if available
            if "response_track" in self.annot:
                gt_rt_start = min(
                    [rf["frame_number"] for rf in self.annot["response_track"]]
                )
                gt_rt_end = max(
                    [rf["frame_number"] for rf in self.annot["response_track"]]
                )
                rt_signal = np.zeros((query_frame,))
                rt_signal[gt_rt_start : gt_rt_end + 1] = 1
                plt.plot(rt_signal, color="green", label="GT response track")
            # Plot peak in signal
            plt.plot(peaks, score_signal_sm[peaks], "rx", label="Peaks")
            save_dir = os.path.join(cfg.logging.save_dir, f"visualizations/{annot_key}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "graph.png")
            plt.savefig(save_path, dpi=500)
            plt.close()
            ###################### Visualize retrievals ########################
            # Visualize crop
            save_path = os.path.join(save_dir, f"visual_crop.png")
            skimage.io.imsave(save_path, visual_crop_im)
            # Visualize retrievals at the peaks
            for peak_idx in peaks:
                peak_images = get_images_at_peak(
                    ret_bboxes, ret_scores, ret_imgs, peak_idx, topk=5
                )
                for image_idx, image in enumerate(peak_images):
                    save_path = os.path.join(
                        save_dir, f"peak_{peak_idx:05d}_rank_{image_idx:03d}.png"
                    )
                    skimage.io.imsave(save_path, image)
            ################## Visualize response track ########################
            save_path = os.path.join(save_dir, f"response_track.mp4")
            writer = imageio.get_writer(save_path)
            for rtf in pred_rt_vis:
                writer.append_data(rtf)
            writer.close()
            ################## Visualize search window #########################
            save_path = os.path.join(save_dir, f"search_window.mp4")
            writer = imageio.get_writer(save_path)
            for sf_ix in range(query_frame):
                writer.append_data(video_reader[sf_ix])
            writer.close()
        # Close video reader
        video_reader.close()

        return {self.key: pred_rts}


class WorkerWithDevice(mp.Process):
    def __init__(self, cfg, task_queue, results_queue, worker_id, device_id):
        self.cfg = cfg
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):

        device = torch.device(f"cuda:{self.device_id}")

        # Create detector
        detectron_cfg = get_detectron_cfg()
        detectron_cfg.merge_from_file(self.cfg.model.config_path)
        detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.cfg.model.score_thresh
        detectron_cfg.MODEL.WEIGHTS = self.cfg.model.checkpoint_path
        detectron_cfg.MODEL.DEVICE = f"cuda:{self.device_id}"
        detectron_cfg.INPUT.FORMAT = "RGB"
        predictor = SiamPredictor(detectron_cfg)

        # Create tracker
        similarity_net = create_similarity_network()
        similarity_net.eval()
        similarity_net.to(device)
        tracker = Tracker(self.cfg, device)

        # Visualization
        os.makedirs(self.cfg.logging.save_dir, exist_ok=True)
        if self.cfg.logging.visualize:
            OmegaConf.save(
                self.cfg, os.path.join(self.cfg.logging.save_dir, "config.yaml")
            )

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except QueueEmpty:
                break
            pred_rts = task.run(predictor, similarity_net, tracker, self.cfg, device)
            if pred_rts is not None:
                results_queue.put(pred_rts)


def perform_vq2d_inference(annotations, cfg):

    num_gpus = torch.cuda.device_count()
    if cfg.data.debug_mode:
        num_gpus = 1
        cfg.data.num_processes_per_gpu = 1

    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for annot in annotations:
        task = Task(annot)
        task_queue.put(task)
    # Results will be stored in this queue
    results_queue = mp.Queue()

    num_processes = cfg.data.num_processes_per_gpu * num_gpus

    pbar = tqdm.tqdm(
        desc=f"Computing VQ2D predictions",
        position=0,
        total=len(annotations),
    )
    workers = [
        WorkerWithDevice(cfg, task_queue, results_queue, i, i % num_gpus)
        for i in range(num_processes)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    predicted_rts = {}
    n_completed = 0
    while n_completed < len(annotations):
        pred = results_queue.get()
        predicted_rts.update(pred)
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()
    return predicted_rts


def convert_annotations_to_list(annotations):
    annotations_list = []
    for v in annotations["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c["clip_uid"]
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    if not q["is_valid"]:
                        continue
                    curr_q = {
                        "metadata": {
                            "video_uid": vuid,
                            "video_start_sec": c["video_start_sec"],
                            "video_end_sec": c["video_end_sec"],
                            "clip_fps": c["clip_fps"],
                            "query_set": qid,
                            "annotation_uid": aid,
                        },
                        "clip_uid": cuid,
                        "query_frame": q["query_frame"],
                        "visual_crop": q["visual_crop"],
                    }
                    if "response_track" in q:
                        curr_q["response_track"] = q["response_track"]
                    annotations_list.append(curr_q)
    return annotations_list


def format_predictions(annotations, predicted_rts):
    # Format predictions
    predictions = {
        "version": annotations["version"],
        "challenge": "ego4d_vq2d_challenge",
        "results": {"videos": []},
    }
    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                apred = {
                    "query_sets": {},
                    "annotation_uid": auid,
                }
                for qid in a["query_sets"].keys():
                    if (auid, qid) in predicted_rts:
                        rt_pred = predicted_rts[(auid, qid)][0].to_json()
                        apred["query_sets"][qid] = rt_pred
                    else:
                        apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    return predictions


@hydra.main(config_path="vq2d", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load annotations
    annot_path = osp.join(cfg.data.annot_root, f"vq_{cfg.data.split}.json")
    with open(annot_path) as fp:
        annotations = json.load(fp)
    annotations_list = convert_annotations_to_list(annotations)

    if cfg.data.debug_mode:
        annotations_list = annotations_list[: cfg.data.debug_count]
    elif cfg.data.subsample:
        annotations_list = annotations_list[::3]
    predicted_rts = perform_vq2d_inference(annotations_list, cfg)
    # Convert prediction to challenge format
    predictions = format_predictions(annotations, predicted_rts)
    with open(cfg.logging.stats_save_path, "w") as fp:
        json.dump(predictions, fp)


if __name__ == "__main__":
    main()

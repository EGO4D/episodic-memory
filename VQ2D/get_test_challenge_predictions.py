import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import time

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
from vq2d.baselines import (
    create_similarity_network,
    get_clip_name_from_clip_uid,
    perform_retrieval,
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
        bbox_images.append(image[bbox.y1 : bbox.y2 + 1, bbox.x1 : bbox.x2 + 1])
    return bbox_images


def predict_vq_test(annotations, cfg, device_id, use_tqdm=False):

    data_cfg = cfg.data
    sig_cfg = cfg.signals

    pred_response_tracks = []

    device = torch.device(f"cuda:{device_id}")

    # Create detector
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.merge_from_file(cfg.model.config_path)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.score_thresh
    detectron_cfg.MODEL.WEIGHTS = cfg.model.checkpoint_path
    detectron_cfg.MODEL.DEVICE = f"cuda:{device_id}"
    detectron_cfg.INPUT.FORMAT = "RGB"
    predictor = SiamPredictor(detectron_cfg)

    # Create tracker
    similarity_net = create_similarity_network()
    similarity_net.eval()
    similarity_net.to(device)
    tracker = Tracker(cfg)

    # Visualization
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    if cfg.logging.visualize:
        OmegaConf.save(cfg, os.path.join(cfg.logging.save_dir, "config.yaml"))

    annotations_iter = tqdm.tqdm(annotations) if use_tqdm else annotations
    for annotation in annotations_iter:
        start_time = time.time()
        clip_uid = annotation["clip_uid"]
        annotation_uid = annotation["metadata"]["annotation_uid"]
        # Load clip from file
        clip_path = os.path.join(
            data_cfg.data_root, get_clip_name_from_clip_uid(clip_uid)
        )
        video_reader = pims.Video(clip_path)
        query_frame = annotation["query_frame"]
        visual_crop = annotation["visual_crop"]
        vcfno = annotation["visual_crop"]["frame_number"]
        clip_frames = video_reader[0 : max(query_frame, vcfno) + 1]
        clip_read_time = time.time() - start_time
        start_time = time.time()
        # Retrieve nearest matches and their scores per image
        ret_bboxes, ret_scores, ret_imgs, visual_crop_im = perform_retrieval(
            clip_frames,
            visual_crop,
            query_frame,
            predictor,
            batch_size=data_cfg.rcnn_batch_size,
            recency_factor=cfg.model.recency_factor,
            subsampling_factor=cfg.model.subsampling_factor,
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
        # Perform tracking to predict response track
        search_frames = clip_frames[: query_frame - 1]
        if len(peaks) > 0:
            init_state = ret_bboxes[peaks[-1]][0]
            init_frame = clip_frames[init_state.fno]
            pred_rt, pred_rt_vis = tracker(
                init_state, init_frame, search_frames, similarity_net, device
            )
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]
            pred_response_tracks.append(pred_rts)
        else:
            pred_rt = [ret_bboxes[-1][0]]
            pred_rt_vis = []
            pred_rts = [ResponseTrack(pred_rt, score=1.0)]
            pred_response_tracks.append(pred_rts)

        tracking_time_taken = time.time() - start_time
        print(
            "====> Data uid: {} | search window :{:>8d} frames | "
            "clip read time: {:>6.2f} mins | "
            "detection time: {:>6.2f} mins | "
            "peak signal time: {:>6.2f} mins | "
            "tracking time: {:>6.2f} mins".format(
                annotation["clip_uid"],
                annotation["query_frame"],
                clip_read_time / 60.0,
                detection_time_taken / 60.0,
                peak_signal_time_taken / 60.0,
                tracking_time_taken / 60.0,
            )
        )

        # Note: This visualization does not work for subsampled evaluation.
        if cfg.logging.visualize:
            ####################### Visualize the peaks ########################
            plt.figure(figsize=(6, 6))
            # Plot raw signals
            # plt.plot(score_signal, color="gray", label="Original signal")
            plt.plot(score_signal_sm, color="blue", label="Similarity scores")
            # Plot highest-scoring pred response track
            pred_rt_start, pred_rt_end = pred_response_tracks[-1][0].temporal_extent
            rt_signal = np.zeros((query_frame,))
            rt_signal[pred_rt_start : pred_rt_end + 1] = 1
            plt.plot(rt_signal, color="red", label="Pred response track")
            # Plot peak in signal
            plt.plot(peaks, score_signal_sm[peaks], "rx", label="Peaks")
            save_path = os.path.join(
                cfg.logging.save_dir, f"example_{annotation_uid}_graph.png"
            )
            plt.savefig(save_path, dpi=500)
            plt.close()
            ###################### Visualize retrievals ########################
            # Visualize crop
            save_path = os.path.join(
                cfg.logging.save_dir, f"example_{annotation_uid}_visual_crop.png"
            )
            skimage.io.imsave(save_path, visual_crop_im)
            # Visualize retrievals at the peaks
            for peak_idx in peaks:
                peak_images = get_images_at_peak(
                    ret_bboxes, ret_scores, ret_imgs, peak_idx, topk=5
                )
                for image_idx, image in enumerate(peak_images):
                    save_path = os.path.join(
                        cfg.logging.save_dir,
                        f"example_{annotation_uid}_peak_{peak_idx:05d}_rank_{image_idx:03d}.png",
                    )
                    skimage.io.imsave(save_path, image)
            ################## Visualize response track ########################
            save_path = os.path.join(cfg.logging.save_dir, f"example_{annotation_uid}_rt.mp4")
            writer = imageio.get_writer(save_path)
            for rtf in pred_rt_vis:
                writer.append_data(rtf)
            writer.close()
            ################## Visualize search window #########################
            save_path = os.path.join(cfg.logging.save_dir, f"example_{annotation_uid}_sw.mp4")
            writer = imageio.get_writer(save_path)
            for sf in search_frames:
                writer.append_data(sf)
            writer.close()

    return pred_response_tracks


def _mp_aux_fn(inputs):
    return predict_vq_test(*inputs)


def predict_vq_test_parallel(annotations, cfg):
    if cfg.data.debug_mode:
        cfg.data.num_processes = 1

    context = mp.get_context("forkserver")
    pool = context.Pool(cfg.data.num_processes, maxtasksperchild=2)
    # Split data across processes
    B = cfg.data.batch_size
    mp_annotations = [annotations[i : (i + B)] for i in range(0, len(annotations), B)]
    N = len(mp_annotations)
    devices = [i for i in range(torch.cuda.device_count())]
    mp_cfgs = [cfg for _ in range(N)]
    mp_devices = [devices[i % len(devices)] for i in range(N)]
    mp_inputs = zip(mp_annotations, mp_cfgs, mp_devices)
    # Perform task
    list_of_outputs = list(tqdm.tqdm(pool.imap(_mp_aux_fn, mp_inputs), total=N))
    predicted_rts = []
    for output in list_of_outputs:
        predicted_rts += output
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
                    if not q['is_valid']:
                        continue
                    annotations_list.append(
                        {
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
                    )
    return annotations_list


def format_predictions(annotations, annotations_list, predicted_rts):
    # Get predictions for each annotation_uid
    annotation_uid_to_prediction = {}
    for pred_rt, annot in zip(predicted_rts, annotations_list):
        auid = annot["metadata"]["annotation_uid"]
        query_set = annot["metadata"]["query_set"]
        if auid not in annotation_uid_to_prediction:
            annotation_uid_to_prediction[auid] = {}
        annotation_uid_to_prediction[auid][query_set] = [rt.to_json() for rt in pred_rt]
    # Format predictions
    predictions = {
        "version": "1.0",
        "challenge": "ego4d_vq2d_challenge",
        "results": {
            "videos": []
        }
    }
    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                if auid in annotation_uid_to_prediction:
                    apred = {
                        "query_sets": annotation_uid_to_prediction[auid][0],
                        'annotation_uid': auid, 
                    }
                else:
                    apred = {
                        "query_sets": {
                            qid: {"bboxes": [], "score": 0.0}
                            for qid in a["query_sets"].keys()
                        }, 
                        'annotation_uid': auid,
                    }
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    return predictions


@hydra.main(config_path="vq2d", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load annotations
    annot_path = osp.join(cfg.data.annot_root, f"vq_test_unannotated.json")
    with open(annot_path) as fp:
        annotations = json.load(fp)
    annotations_list = convert_annotations_to_list(annotations)

    if cfg.data.debug_mode:
        annotations_list = annotations_list[: cfg.data.debug_count]
    elif cfg.data.subsample:
        annotations_list = annotations_list[::3]
    predicted_rts = predict_vq_test_parallel(annotations_list, cfg)
    # Convert prediction to challenge format
    predictions = format_predictions(annotations, annotations_list, predicted_rts)
    with open(cfg.logging.stats_save_path, "w") as fp:
        json.dump(predictions, fp)


if __name__ == "__main__":
    main()

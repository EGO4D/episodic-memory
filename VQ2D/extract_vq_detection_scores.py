import json
import os
import os.path as osp
from queue import Empty as QueueEmpty
from typing import Any, Dict

import cv2
import pims
import torch
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2_extensions.config import get_cfg as get_detectron_cfg
from einops import rearrange, asnumpy
from torch import multiprocessing as mp
from vq2d.baselines import (
    get_clip_name_from_clip_uid,
    SiamPredictor,
)
from vq2d.baselines.utils import extract_window_with_context
from vq2d.structures import BBox

setup_logger()

import hydra
from omegaconf import DictConfig


def extract_clip_bboxes_and_scores(
    clip_reader,
    clip_path,
    visual_crop: Dict[str, Any],
    query_frame: int,
    net: DefaultPredictor,
    batch_size: int = 8,
    key_name: str = None,
    downscale_height: int = 700,
):
    """
    Given a visual crop and frames from a clip, retrieve the bounding box proposal
    from each frame that is most similar to the visual crop.
    """
    vc_fno = visual_crop["frame_number"]
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]

    # Load visual crop frame
    if vc_fno >= len(clip_reader):
        print(
            "=====> WARNING: Going out of range. Clip path: {}, Len: {}, j: {}".format(
                clip_path, len(clip_reader), vc_fno
            )
        )
    reference = clip_reader[vc_fno]  # RGB format
    ## Resize visual crop if stored aspect ratio was incorrect
    if (reference.shape[0] != oheight) or (reference.shape[1] != owidth):
        reference = cv2.resize(reference, (owidth, oheight))
    reference = torch.as_tensor(rearrange(reference, "h w c -> () c h w"))
    reference = reference.float()
    ref_bbox = (
        visual_crop["x"],
        visual_crop["y"],
        visual_crop["x"] + visual_crop["width"],
        visual_crop["y"] + visual_crop["height"],
    )
    reference = extract_window_with_context(
        reference,
        ref_bbox,
        net.cfg.INPUT.REFERENCE_CONTEXT_PAD,
        size=net.cfg.INPUT.REFERENCE_SIZE,
        pad_value=125,
    )
    reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
    # Define search window
    search_window = list(range(0, query_frame))

    # Load reference frames and perform detection
    ret_bboxes = []
    ret_scores = []
    # Batch extract predictions
    desc = None if key_name is None else f"Processing {key_name}"
    for i in tqdm.tqdm(
        range(0, len(search_window), batch_size), desc=desc, position=1, leave=False
    ):
        bimages = []
        breferences = []
        image_scales = []
        i_end = min(i + batch_size, len(search_window))
        for j in range(i, i_end):
            if search_window[j] >= len(clip_reader):
                print(
                    "=====> WARNING: Going out of range. Clip path: {}, Len: {}, j: {}".format(
                        clip_path, len(clip_reader), j
                    )
                )
            image = clip_reader[search_window[j]]  # RGB format
            if image.shape[:2] != (oheight, owidth):
                image = cv2.resize(image, (owidth, oheight))
            # Scale-down image to reduce memory consumption
            image_scale = float(downscale_height) / image.shape[0]
            image = cv2.resize(image, None, fx=image_scale, fy=image_scale)
            bimages.append(image)
            breferences.append(reference)
            image_scales.append(image_scale)
        # Perform inference
        all_outputs = net(bimages, breferences)
        # Unpack outputs
        for j in range(i, i_end):
            instances = all_outputs[j - i]["instances"]
            image_scale = image_scales[j - i]
            # Re-scale bboxes
            ret_bbs = (
                asnumpy(instances.pred_boxes.tensor / image_scale).astype(int).tolist()
            )
            ret_bbs = [BBox(search_window[j], *bbox) for bbox in ret_bbs]
            ret_scs = asnumpy(instances.scores).tolist()
            ret_bboxes.append(ret_bbs)
            ret_scores.append(ret_scs)
        del all_outputs

    return ret_bboxes, ret_scores


class Task:
    def __init__(self, annots):
        super().__init__()
        self.annots = annots
        # Ensure that all annotations belong to the same clip
        clip_uid = annots[0]["clip_uid"]
        for annot in self.annots:
            assert annot["clip_uid"] == clip_uid

    def run(self, predictor, cfg):
        data_cfg = cfg.data
        clip_uid = self.annots[0]["clip_uid"]
        if clip_uid is None:
            return None
        # Load clip from file
        clip_path = os.path.join(
            data_cfg.data_root, get_clip_name_from_clip_uid(clip_uid)
        )
        video_reader = pims.Video(clip_path)
        for annot in self.annots:
            annotation_uid = annot["metadata"]["annotation_uid"]
            query_set_id = annot["metadata"]["query_set"]
            key_name = f"{annotation_uid}_{query_set_id}"
            save_path = os.path.join(cfg.model.cache_root, f"{key_name}.pt")
            if not os.path.isfile(save_path):
                query_frame = annot["query_frame"]
                visual_crop = annot["visual_crop"]
                # Retrieve nearest matches and their scores per image
                ret_bboxes, ret_scores = extract_clip_bboxes_and_scores(
                    video_reader,
                    clip_path,
                    visual_crop,
                    query_frame,
                    predictor,
                    batch_size=data_cfg.rcnn_batch_size,
                    key_name=key_name,
                )
                torch.save(
                    {"ret_bboxes": ret_bboxes, "ret_scores": ret_scores}, save_path
                )
        video_reader.close()

        return key_name


class WorkerWithDevice(mp.Process):
    def __init__(self, cfg, task_queue, results_queue, worker_id, device_id):
        self.cfg = cfg
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):
        # Create detector
        detectron_cfg = get_detectron_cfg()
        detectron_cfg.set_new_allowed(True)
        detectron_cfg.merge_from_file(self.cfg.model.config_path)
        detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.cfg.model.score_thresh
        detectron_cfg.MODEL.WEIGHTS = self.cfg.model.checkpoint_path
        detectron_cfg.MODEL.DEVICE = f"cuda:{self.device_id}"
        detectron_cfg.INPUT.FORMAT = "RGB"
        predictor = SiamPredictor(detectron_cfg)

        os.makedirs(self.cfg.logging.save_dir, exist_ok=True)

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except QueueEmpty:
                break
            key_name = task.run(predictor, self.cfg)
            results_queue.put(key_name)
            del task

        del predictor


def extract_detection_scores(annotations, cfg):
    if cfg.data.debug_mode:
        cfg.data.num_processes_per_gpu = 1

    num_gpus = torch.cuda.device_count()

    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for _, annots in annotations.items():
        task = Task(annots)
        task_queue.put(task)
    # Results will be stored in this queue
    results_queue = mp.Queue()

    num_processes = cfg.data.num_processes_per_gpu * num_gpus

    pbar = tqdm.tqdm(
        desc=f"Extracting detection scores",
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
    n_completed = 0
    while n_completed < len(annotations):
        _ = results_queue.get()
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()


def convert_annotations_to_clipwise_list(annotations):
    clipwise_annotations_list = {}
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
                    if cuid not in clipwise_annotations_list:
                        clipwise_annotations_list[cuid] = []
                    clipwise_annotations_list[cuid].append(curr_q)
    return clipwise_annotations_list


@hydra.main(config_path="vq2d", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load annotations
    annot_path = osp.join(cfg.data.annot_root, f"vq_{cfg.data.split}.json")
    with open(annot_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = convert_annotations_to_clipwise_list(annotations)

    assert cfg.model.cache_root != ""
    os.makedirs(cfg.model.cache_root, exist_ok=True)

    if cfg.data.debug_mode or cfg.data.subsample:
        clips_list = list(clipwise_annotations_list.keys())
        # Filter None clip
        clips_list = sorted([c for c in clips_list if c is not None])
        if cfg.data.debug_mode:
            clips_list = clips_list[: cfg.data.debug_count]
        elif cfg.data.subsample:
            clips_list = clips_list[::3]
        clipwise_annotations_list = {
            k: clipwise_annotations_list[k] for k in clips_list
        }

    extract_detection_scores(clipwise_annotations_list, cfg)


if __name__ == "__main__":
    main()

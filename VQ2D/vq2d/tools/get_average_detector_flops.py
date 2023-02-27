import typing

import argparse
import cv2
import glob
import imageio
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import tqdm

from collections import defaultdict
from detectron2_extensions.config import get_cfg
from deepspeed.profiling.flops_profiler import get_model_profile
from einops import rearrange, asnumpy
from vq2d.baselines import SiamPredictor
from vq2d.baselines.utils import extract_window_with_context
from vq2d.baselines.utils import get_image_name_from_clip_uid


def setup_model(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.WEIGHTS = args.pretrained_weights
    cfg.MODEL.DEVICE = f"cuda:{args.gpu_id}"
    cfg.INPUT.FORMAT = "RGB"
    cfg.freeze()
    model = SiamPredictor(cfg)
    return model


def process_annotations(annotations):
    proc_annotations = []
    for v in annotations["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c["clip_uid"]
            if cuid is None:
                continue
            for a in c["annotations"]:
                for qid, q in a["query_sets"].items():
                    if not q["is_valid"]:
                        continue
                    vc_fno = q["visual_crop"]["frame_number"]
                    vc_path = os.path.join(
                        f"data/images", get_image_name_from_clip_uid(cuid, vc_fno)
                    )
                    if not os.path.isfile(vc_path):
                        continue
                    proc_annotations.append(
                        {
                            "video_uid": vuid,
                            "clip_uid": cuid,
                            "query_set_id": qid,
                            "object_title": q["object_title"],
                            "query_frame": q["query_frame"],
                            "visual_crop": q["visual_crop"],
                        }
                    )
    return proc_annotations


def get_visual_crop_image(reference, visual_crop, cfg):
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]

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
        cfg.INPUT.REFERENCE_CONTEXT_PAD,
        size=cfg.INPUT.REFERENCE_SIZE,
        pad_value=125,
    )
    reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
    return reference


def process_image(image, visual_crop, downscale_height=700):
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]
    if image.shape[:2] != (oheight, owidth):
        image = cv2.resize(image, (owidth, oheight))
    # Scale-down image to reduce memory consumption
    image_scale = float(downscale_height) / image.shape[0]
    image = cv2.resize(image, None, fx=image_scale, fy=image_scale)
    return image


def compute_average_flops(args):
    with open(args.annotations, "r") as fp:
        annotations = json.load(fp)
    annotations = process_annotations(annotations)
    annotations = sorted(
        annotations,
        key=lambda x: (x["video_uid"], x["clip_uid"], x["query_set_id"], x["object_title"]),
    )
    # Sample a random subset of annotations
    random.shuffle(annotations)
    annotations = annotations[:10]
    # Setup model
    model = setup_model(args)
    # Calculate FLOPs
    total_count = 0
    total_flops = 0
    total_macs = 0
    total_params = 0
    for annot in tqdm.tqdm(annotations, desc="Computing average FLOPs"):
        clip_uid = annot["clip_uid"]
        visual_crop = annot["visual_crop"]
        vcfno = visual_crop["frame_number"]
        # Load visual crop image
        reference_image_path = os.path.join(
            f"data/images", get_image_name_from_clip_uid(clip_uid, vcfno)
        )
        reference_image = imageio.imread(reference_image_path)
        reference = get_visual_crop_image(reference_image, visual_crop, model.cfg)
        if model.input_format == "RGB":
            reference = reference[:, :, ::-1]
        reference = torch.as_tensor(reference.astype("float32").transpose(2, 0, 1))
        # Sample random images from clips
        image_paths = sorted(glob.glob(os.path.join(f"data/images/{clip_uid}/*.png")))
        random.shuffle(image_paths)
        for image_path in image_paths[:10]:
            try:
                image = imageio.imread(image_path)
            except ValueError:
                continue
            image = process_image(image, visual_crop)
            with torch.no_grad():
                if model.input_format == "RGB":
                    image = image[:, :, ::-1]
                height, width = image.shape[:2]
                image = model.aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = [
                    {
                        "image": image,
                        "reference": reference,
                        "height": height,
                        "width": width,
                    }
                ]
                flops, macs, params = get_model_profile(
                    model.model,
                    args=(inputs,),
                    print_profile=False,
                    detailed=False,
                    as_string=False,
                )
                total_count += 1
                total_flops += float(flops) / 1e9
                total_macs += float(macs) / 1e9
                total_params += float(params) / 1e6

    print(f"=======> Average stats over {total_count} images")
    print("{:<20s} | {:>9.3f}".format("Total GFLOPs", total_flops / total_count))
    print("{:<20s} | {:>9.3f}".format("Total GMACs", total_macs / total_count))
    print("{:<20s} | {:>9.3f}".format("Total params (M)", total_params / total_count))
    print("-----------------------------------------------")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        default="pretrained_models/siam_rcnn_residual/config.yaml",
        type=str,
    )
    parser.add_argument(
        "--pretrained_weights",
        default="pretrained_models/siam_rcnn_residual/model.pth",
        type=str
    )
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--annotations", default="data/vq_val.json", type=str)

    args = parser.parse_args()

    random.seed(123)
    compute_average_flops(args)
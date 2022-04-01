import copy
import gzip
import json
import os
import os.path as osp
import random
from typing import Any, Optional, Sequence, List, Dict

import imagesize
import numpy as np
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from ..constants import DATASET_FILE_TEMPLATE
from .utils import (
    get_image_name_from_clip_uid,
    get_bbox_from_data,
    get_image_id_from_data,
)


def _register_visual_query_dataset(
    data_id: str,
    data_path: str,
    images_root: str,
    **kwargs: Any,
) -> None:
    """Helper function to register visual query datasets."""

    def visual_query_dataset_function():
        return visual_query_dataset(data_path, images_root, **kwargs)

    try:
        DatasetCatalog.register(data_id, visual_query_dataset_function)
    except AssertionError:
        # Skip this step if it is already registered
        pass

    MetadataCatalog.get(data_id).thing_classes = ["visual_crop"]
    MetadataCatalog.get(data_id).thing_dataset_id_to_contiguous_id = {0: 0}


def register_visual_query_datasets(
    data_root: str,
    images_root: str,
    data_key: str,
    bbox_aspect_scale: Optional[float] = None,
    bbox_area_scale: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Given dataset paths and other configuration arguments, it registers a
    visual query dataset with the specified arguments.
    """
    for split in ["train", "val", "test"]:
        data_path = osp.join(data_root, DATASET_FILE_TEMPLATE.format(split))
        bbox_aspect_ratio_range = None
        bbox_area_ratio_range = None
        data_id = f"{data_key}_{split}"
        if bbox_aspect_scale is not None:
            assert 0.0 < bbox_aspect_scale < 1.0
            bbox_aspect_ratio_range = (bbox_aspect_scale, 1.0 / bbox_aspect_scale)
        if bbox_area_scale is not None:
            assert 0.0 < bbox_area_scale < 1.0
            bbox_area_ratio_range = (bbox_area_scale, 1.0 / bbox_area_scale)
        _register_visual_query_dataset(
            data_id,
            data_path,
            images_root,
            bbox_aspect_ratio_range=bbox_aspect_ratio_range,
            bbox_area_ratio_range=bbox_area_ratio_range,
            **kwargs,
        )


def visual_query_dataset(
    data_path: str,
    images_root: str,
    bbox_aspect_ratio_range: Optional[Sequence[float]] = None,
    bbox_area_ratio_range: Optional[Sequence[float]] = None,
    perform_response_augmentation: bool = False,
    augmentation_limit: bool = 10,
) -> List[Dict[str, Any]]:
    with gzip.open(data_path, "rt") as fp:
        annotations = json.load(fp)
    data_samples = []
    for annot_ix, annot in tqdm.tqdm(enumerate(annotations), total=len(annotations)):
        clip_uid = annot["clip_uid"]
        vc_bbox = get_bbox_from_data(annot["visual_crop"])
        vc_fno = annot["visual_crop"]["frame_number"]
        vc_path = get_image_name_from_clip_uid(clip_uid, vc_fno)
        vc_path = osp.join(images_root, vc_path)
        if not os.path.isfile(vc_path):
            continue
        # Is aspect ratio correction needed?
        actual_width, actual_height = imagesize.get(vc_path)
        vc_width = annot["visual_crop"]["original_width"]
        vc_height = annot["visual_crop"]["original_height"]
        vc_arc = False
        if (vc_width, vc_height) != (actual_width, actual_height):
            vc_arc = True
            print("=======> VC needs aspect ratio correction")
        # Sort response track by frame number
        annot["response_track"] = sorted(
            annot["response_track"], key=lambda x: x["frame_number"]
        )
        # Get aspect ratio for the largest response track BBoxes
        bbox_areas = []
        aspect_ratios = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            bba = rf_data["width"] * rf_data["height"]
            ar = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            bbox_areas.append(bba)
            aspect_ratios.append(ar)
        bbox_areas = np.array(bbox_areas)
        aspect_ratios = np.array(aspect_ratios)
        bbox_idxs = np.argsort(-bbox_areas)[:5]
        std_bbox_area = np.median(bbox_areas[bbox_idxs]).item()
        std_aspect_ratio = np.median(aspect_ratios[bbox_idxs]).item()
        # Create one sample for every (visual query, response frame) pairs
        curr_data_samples = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            rf_bbox = get_bbox_from_data(rf_data)
            rf_fno = rf_data["frame_number"]
            rf_path = get_image_name_from_clip_uid(clip_uid, rf_fno)
            rf_path = osp.join(images_root, rf_path)
            if not os.path.isfile(rf_path):
                continue
            # NOTE: By default, the category_id will be 0 always. This is because there is only
            # one class, corresponding to the right match. Within detectron2, the unmatched
            # bbox proposals will automatically be set to 1, the background class.
            category_id = 0
            # Is aspect ratio correction needed?
            actual_width, actual_height = imagesize.get(rf_path)
            rf_width = rf_data["original_width"]
            rf_height = rf_data["original_height"]
            response_arc = False
            if (rf_width, rf_height) != (actual_width, actual_height):
                response_arc = True
                print("=======> RF needs aspect ratio correction")
            # Clean dataset
            bbox_area = rf_data["width"] * rf_data["height"]
            aspect_ratio = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            clean = True
            if clean and (bbox_aspect_ratio_range is not None):
                ratio = aspect_ratio / (std_aspect_ratio + 1e-10)
                clean = (
                    bbox_aspect_ratio_range[0] <= ratio <= bbox_aspect_ratio_range[1]
                )
            if clean and (bbox_area_ratio_range is not None):
                ratio = bbox_area / (std_bbox_area + 1e-10)
                clean = bbox_area_ratio_range[0] <= ratio <= bbox_area_ratio_range[1]
            if not clean:
                continue

            curr_data_samples.append(
                {
                    "image_id": get_image_id_from_data(annot, annot_ix, rf_idx),
                    "file_name": rf_path,
                    "info": {
                        "aspect_ratio": aspect_ratio,
                        "bbox_area": bbox_area,
                        "std_aspect_ratio": std_aspect_ratio,
                        "std_bbox_area": std_bbox_area,
                    },
                    "width": rf_width,
                    "height": rf_height,
                    "incorrect_aspect_ratio": response_arc,
                    "annotations": [
                        {
                            "bbox": rf_bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": category_id,
                        }
                    ],
                    "reference": {
                        "file_name": vc_path,
                        "width": vc_width,
                        "height": vc_height,
                        "incorrect_aspect_ratio": vc_arc,
                        "annotations": [
                            {
                                "bbox": vc_bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "textual_name": annot["object_title"],
                            }
                        ],
                    },
                }
            )
        data_samples += curr_data_samples

        # Optionally, augment dataset by creating (response frame, response frame) pairs
        if not perform_response_augmentation:
            continue
        ## Get a list of good response frames to serve as dummy visual queries.
        ## A good response frame is one that is clean according to bbox ratio criteria.
        clean_response_frames = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            rf_bbox = get_bbox_from_data(rf_data)
            rf_fno = rf_data["frame_number"]
            rf_path = get_image_name_from_clip_uid(clip_uid, rf_fno)
            rf_path = osp.join(images_root, rf_path)
            if not os.path.isfile(rf_path):
                continue
            # Is aspect ratio correction needed?
            actual_width, actual_height = imagesize.get(rf_path)
            rf_width = rf_data["original_width"]
            rf_height = rf_data["original_height"]
            response_arc = False
            if (rf_width, rf_height) != (actual_width, actual_height):
                response_arc = True
                print("=======> RF needs aspect ratio correction")
            bbox_area = rf_data["width"] * rf_data["height"]
            aspect_ratio = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            clean = True
            if clean:
                ratio = aspect_ratio / (std_aspect_ratio + 1e-10)
                clean = 0.85 <= ratio <= (1.0 / 0.85)
            if clean:
                ratio = bbox_area / (std_bbox_area + 1e-10)
                clean = 0.50 <= ratio <= (1.0 / 0.5)
            if not clean:
                continue

            clean_response_frames.append(
                {
                    "file_name": rf_path,
                    "width": rf_width,
                    "height": rf_height,
                    "incorrect_aspect_ratio": response_arc,
                    "annotations": [
                        {
                            "bbox": rf_bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "textual_name": annot["object_title"],
                        }
                    ],
                }
            )
        random.shuffle(clean_response_frames)
        clean_response_frames = clean_response_frames[:augmentation_limit]

        ## Add new data with augmented samples
        for ds in curr_data_samples:
            for crf in clean_response_frames:
                ds = copy.deepcopy(ds)
                ds["reference"] = crf
                data_samples.append(ds)

    return data_samples

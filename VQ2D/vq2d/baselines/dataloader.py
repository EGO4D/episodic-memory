import copy
import logging
from typing import List, Union

import cv2
import numpy as np
import torch
from detectron2.config import CfgNode, configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation

from .utils import extract_window_with_context


def build_augmentation(
    cfg: CfgNode, is_train: bool, mode: str = "input"
) -> List[Augmentation]:
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
        mode - can be input / reference
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    if is_train and mode == "input" and cfg.INPUT.ENABLE_RANDOM_ROTATION:
        augmentation.append(
            T.RandomRotation(
                cfg.INPUT.ROTATION_ANGLES,
                expand=cfg.INPUT.ROTATION_EXPAND,
            )
        )
    elif (
        is_train
        and mode == "reference"
        and cfg.INPUT.ENABLE_RANDOM_ROTATION_VISUAL_CROP
    ):
        augmentation.append(
            T.RandomRotation(
                cfg.INPUT.ROTATION_ANGLES,
                expand=cfg.INPUT.ROTATION_EXPAND,
            )
        )

    if is_train and cfg.INPUT.ENABLE_RANDOM_BRIGHTNESS:
        augmentation.append(
            T.RandomBrightness(*cfg.INPUT.RANDOM_BRIGHTNESS_VALS),
        )

    return augmentation


class VisualQueryDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model. This is modified to load
    the data for visual queries task.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Read the visual crop from dataset_dict["reference"]["file_name"]
    3. Applies cropping/geometric transforms to the image and annotations
    4. Crop out the query object from the visual crop image
    5. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_ref: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        reference_size: int,
        reference_context_pad: int,
        transform_reference: bool,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.augmentations_ref      = T.AugmentationList(augmentations_ref)
        self.image_format           = image_format
        self.reference_size         = reference_size
        self.reference_context_pad  = reference_context_pad
        self.transform_reference    = transform_reference
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[VisualQueryDatasetMapper] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train, mode="input")
        ref_augs = build_augmentation(cfg, is_train, mode="reference")

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_ref": ref_augs,
            "image_format": cfg.INPUT.FORMAT,
            "reference_size": cfg.INPUT.REFERENCE_SIZE,
            "reference_context_pad": cfg.INPUT.REFERENCE_CONTEXT_PAD,
            "transform_reference": cfg.INPUT.TRANSFORM_VISUAL_CROP,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept

        NOTE: Currently, this call assumes that the annotations are consistent with
        the sizes of the image loaded. For example, the bounding box should have
        been annotated on the same image that is loaded. Any inconsistency here
        can lead to errors down the line.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # Correct aspect ratio if incorrect
        if dataset_dict["incorrect_aspect_ratio"]:
            image = cv2.resize(image, (dataset_dict["width"], dataset_dict["height"]))
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # process reference visual crop
        ref_dict = dataset_dict["reference"]
        ref_file_name = ref_dict["file_name"]
        reference = utils.read_image(ref_file_name, format=self.image_format)
        # Correct aspect ratio if incorrect
        if ref_dict["incorrect_aspect_ratio"]:
            reference = cv2.resize(reference, (ref_dict["width"], ref_dict["height"]))
        utils.check_image_size(ref_dict, reference)
        if self.transform_reference:
            aug_ref = T.AugInput(reference)
            transforms_ref = self.augmentations_ref(aug_ref)
            reference = aug_ref.image

            reference_shape = reference.shape[:2]  # h, w

            ref_annot = utils.transform_instance_annotations(
                ref_dict["annotations"][0], transforms_ref, reference_shape
            )
            ref_dict["annotations"][0]["bbox"] = ref_annot["bbox"]

        reference = torch.as_tensor(np.ascontiguousarray(reference.transpose(2, 0, 1)))
        reference = reference.unsqueeze(0).float()
        ref_bbox = ref_dict["annotations"][0]["bbox"]
        reference = extract_window_with_context(
            reference,
            ref_bbox,
            p=self.reference_context_pad,
            size=self.reference_size,
            pad_value=125,
        )
        dataset_dict["reference"] = reference.squeeze(0).byte()

        return dataset_dict

from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tmodels
from einops import rearrange

Numeric = Union[int, float]

from ..constants import (
    CLIP_NAME_PATTERN,
    IMAGE_NAME_PATTERN,
)
from ..structures import BBox


def convert_image_np2torch(image: np.ndarray) -> torch.Tensor:
    """
    image - (H, W, 3) numpy array
    """
    mean = torch.Tensor([[[0.485, 0.456, 0.406]]])
    std = torch.Tensor([[[0.229, 0.224, 0.225]]])
    image = torch.from_numpy(image).float() / 255.0
    image = (image - mean) / std
    image = rearrange(image, "h w c -> () c h w")
    return image


def convert_annot_to_bbox(annot: Dict[str, Any]) -> BBox:
    return BBox(
        annot["frame_number"],
        annot["x"],
        annot["y"],
        annot["x"] + annot["width"],
        annot["y"] + annot["height"],
    )


def get_clip_name_from_clip_uid(clip_uid: str) -> str:
    return CLIP_NAME_PATTERN.format(clip_uid)


def get_image_name_from_clip_uid(clip_uid: str, fno: int) -> str:
    return IMAGE_NAME_PATTERN.format(clip_uid, fno + 1)


def create_similarity_network(pretrained: bool = True) -> nn.Sequential:
    resnet50 = tmodels.resnet50(pretrained=pretrained)
    net = nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        resnet50.layer4,
        resnet50.avgpool,
        nn.Flatten(),
    )

    return net


def extract_window_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.

    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    x1 = max(x1 - p, 0)
    y1 = max(y1 - p, 0)
    x2 = min(x2 + p, W)
    y2 = min(y2 + p, H)
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # Zero pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = nn.functional.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = nn.functional.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    )

    return window


def get_bbox_from_data(data: Dict[str, Any]) -> List[Numeric]:
    return [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]


def get_image_id_from_data(data: Dict[str, Any], data_ix: int, rno: int) -> str:
    """
    Defines a unique image id for a given VQ data point.
    """
    clip_uid = data["clip_uid"]
    qset = data["query_set"]
    return f"clip-uid_{clip_uid}_idx_{data_ix}_query-set_{qset}_response-idx_{rno}"


def resize_if_needed(image: np.uint8, shape: Tuple[int, int]) -> np.ndarray:
    """
    shape - (width, height) tuple
    """
    width, height = shape
    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    return image

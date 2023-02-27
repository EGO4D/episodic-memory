from typing import Any

import cv2
import numpy as np
import torch
from tracking.models.siammask_tracker.anchors import Anchors


def get_subwindow(im, pos, original_sz, model_sz, avg_chans, out_mode="torch"):
    """
    Crop subwindow region from the whole image.

    This function is called when cropping template / search regions from
    template / search images.
    Do padding if the region is out of the image, then resize the subwindow to a given size.

    Args:
        im (np.array): input image, shape [H, W, 3]
        pos (list): position of the subwindow center.
        original_sz: size of the subwindow.
        model_sz (int): size of model input.
        avg_chans (np.array): average BGR values (3 elements), for padding purpose
    Returns:
        A image patch in tensor format.
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + original_sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + original_sz - 1
    left_pad = int(max(0.0, -context_xmin))
    top_pad = int(max(0.0, -context_ymin))
    right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros(
            (r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8
        )
        te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad :, :] = avg_chans
        im_patch_original = te_im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]
    else:
        im_patch_original = im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch) if out_mode in "torch" else im_patch


def generate_anchor(cfg, score_size):
    """
    Generate anchors according to config.

    Anchor config follows the format:
    anchor_cfg = {
        "stride": 8,
        "ratios": [0.33, 0.5, 1, 2, 3],
        "scales": [8],
        "round_dight": 0,
    }
    """
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * total_stride
    xx, yy = np.meshgrid(
        [ori + total_stride * dx for dx in range(score_size)],
        [ori + total_stride * dy for dy in range(score_size)],
    )
    xx, yy = (
        np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
        np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
    )
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def to_torch(ndarray: Any) -> Any:
    """
    Convert numpy array to pytorch tensor.
    """
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def im_to_torch(img: Any) -> Any:
    """
    Convert cv2 image to pytorch tensor: shape [HWC] -> [CHW].
    """
    img = np.transpose(img, (2, 0, 1))  # H*W*C --> C*H*W
    img = to_torch(img).float()
    return img

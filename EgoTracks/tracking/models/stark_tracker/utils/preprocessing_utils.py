import logging
import time

import cv2

import numpy as np
import torch.nn.functional as F

"""modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later"""


def sample_target(im, pos, crop_sz, output_sz=None, mask=None, verbose=False):
    """Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area
    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        crop_sz - Crop size on the original input image
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        verbose - (bool) Boolean flag that indicates whether to log profiling information.
    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if isinstance(pos, float):
        pos = [pos, pos]

    c = (crop_sz + 1) / 2
    x1 = round(pos[0] - c)
    y1 = round(pos[1] - c)

    if crop_sz < 1:
        raise Exception(f"Too small bounding box. pos {pos}, crop_sz {crop_sz}")

    x2 = x1 + crop_sz

    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad]

    # Pad
    t = time.time()
    im_crop_padded = cv2.copyMakeBorder(
        im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT
    )
    if verbose:
        logging.error(f"padd takes {time.time() - t}")
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    t = time.time()
    att_mask = np.ones((H, W))
    if verbose:
        logging.error(f"Init att_mask {(H, W)} takes {time.time() - t}")
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(
            mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode="constant", value=0
        )

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        t = time.time()
        prev_shape = im_crop_padded.shape
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if verbose:
            logging.error(f"Resize {prev_shape} to {output_sz} {time.time() - t}")
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = F.interpolate(
            mask_crop_padded[None, None],
            (output_sz, output_sz),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

from typing import Any, Dict, Sequence, List

import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from einops import rearrange, asnumpy

from ..structures import BBox
from .utils import extract_window_with_context


def perform_retrieval(
    video_reader: Any,
    visual_crop: Dict[str, Any],
    query_frame: int,
    cached_bboxes: List[BBox],
    cached_scores: List[float],
    recency_factor: float = 1.0,  # Search only within the most recent frames.
    subsampling_factor: float = 1.0,  # Search only within a subsampled set of frames.
    visualize: bool = False,
    reference_pad: int = 16,
    reference_size: int = 256,
):
    """
    Given a visual crop and frames from a clip, retrieve the bounding box proposal
    from each frame that is most similar to the visual crop.
    """
    vc_fno = visual_crop["frame_number"]
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]

    # Load visual crop frame
    reference = video_reader[vc_fno]  # RGB format
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
        reference_pad,
        size=reference_size,
        pad_value=125,
    )
    reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
    # Define search window
    search_window = list(range(0, query_frame))
    ## Pick recent k% of frames
    window_size = int(round(len(search_window) * recency_factor))
    if len(search_window[-window_size:]) > 0:
        search_window = search_window[-window_size:]
    ## Subsample only k% of frames
    window_size = len(search_window)
    idxs_to_sample = np.linspace(
        0, window_size - 1, int(subsampling_factor * window_size)
    ).astype(int)
    if len(idxs_to_sample) > 0:
        search_window = [search_window[i] for i in idxs_to_sample]

    # Gather predictions
    ret_bboxes = [cached_bboxes[s] for s in search_window]
    ret_scores = [cached_scores[s] for s in search_window]
    ret_imgs = (
        [cv2.resize(video_reader[s], (owidth, oheight)) for s in search_window]
        if visualize
        else []
    )
    return ret_bboxes, ret_scores, ret_imgs, reference

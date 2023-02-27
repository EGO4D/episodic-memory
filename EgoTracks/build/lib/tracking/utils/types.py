from dataclasses import dataclass, field
from typing import Dict

from tracking.models.stark_tracker.config.stark_st2.config import STARKParams


@dataclass
class SiamMaskTestParams:
    # SiamMask anchor config
    anchor_cfg: Dict = field(
        default_factory=lambda: {
            "stride": 8,
            "ratios": [0.33, 0.5, 1, 2, 3],
            "scales": [8],
            "round_dight": 0,
        }
    )
    template_size: int = 127
    search_size: int = 255
    confidence_thresh: float = 0.0
    out_size: int = 127
    base_size: int = 8
    total_stride: int = 8
    seg_thr: float = 0.3
    context_amount: float = 0.5
    # This is different from original work, original 0.09
    penalty_k: float = 0.09
    windowing: str = "cosine"
    # This is different from original work, original 0.39
    window_influence: float = 0.39
    # This is different from original work, original 0.38
    lr: float = 0.38
    # If the input image needs to be cropped to smaller patches,
    # this variable controls how many these patches we can send at a time.
    test_batch_size: int = 8
    enable_mask: bool = False
    enable_refine: bool = False


@dataclass
class SiamMaskParams:
    test: SiamMaskTestParams = SiamMaskTestParams()


@dataclass
class Params:
    # 0 for train, 1 for eval
    run_type: int = 1
    # basics
    workers: int = 8
    print_freq: int = 10
    num_shards: int = 1
    num_gpus_per_node: int = 2
    dist_backend: str = "nccl"
    seed: int = -1

    # data related
    annotation_path: str = "manifold://tracking/tree/ego4d/v1/annotations/vq_val.json"
    clip_dir: str = "manifold://tracking/tree/ego4d/clip"
    video_dir: str = (
        "manifold://ego4d_fair/tree/intermediate/canonical/v7/full_scale/canonical"
    )
    model_type: str = "STARK"
    model_path: str = "manifold://tracking/tree/models/STARK/STARKST_ep0050.pth"
    result_dir: str = (
        "manifold://tracking/tree/users/haotang/model_output/stark_train_test"
    )
    config_files: str = ""
    visualize: int = 0
    # is_read_5FPS_clip indicates whether we read frames from the original 30FPS video
    # or 5FPS clip.
    # return_5FPS_frames is only used if is_read_5FPS_clip is False.
    # If return_5FPS_frames is True, we only return frames at 5FPS, but those frames
    # are extracted from the 30FPS video.
    is_read_5FPS_clip: bool = True
    return_5FPS_frames: bool = True
    """
    parameter to control where to track from
    first_bbox: track from the first frame of the video sequence.
                 This assumes the GT bbox for the first frame is given.
    largest_bbox: track from the frame that has the largest GT bounding box.
                    It will run the tracker forward until the end and
                    backward until the beginning of the video sequence
    """
    track_mode: str = "first_bbox"

    # checkpoint-related
    checkpoint_dir: str = ""
    pretrained: str = ""
    resume: str = ""

    # Model parameters
    SiamMask: SiamMaskParams = SiamMaskParams()
    STARK: STARKParams = STARKParams()

    # whether only tracking locally
    is_search_local: bool = True

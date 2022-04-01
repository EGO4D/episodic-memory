from detectron2.config import CfgNode
from detectron2.config import get_cfg as get_default_cfg


# Extend detectron2 defaults
_C = get_default_cfg()


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
# Image augmentations for SiamRCNN
_C.INPUT.ENABLE_RANDOM_BRIGHTNESS = False
_C.INPUT.RANDOM_BRIGHTNESS_VALS = [0.75, 1.33333]
_C.INPUT.ROTATION_ANGLES = [-60, 60]
_C.INPUT.ROTATION_EXPAND = True
_C.INPUT.ENABLE_RANDOM_ROTATION = False
_C.INPUT.TRANSFORM_VISUAL_CROP = False
_C.INPUT.ENABLE_RANDOM_ROTATION_VISUAL_CROP = False
# Visual crop generation
_C.INPUT.REFERENCE_CONTEXT_PAD = 16  # Pixel padding around visual crop
_C.INPUT.REFERENCE_SIZE = 256  # Visual crop size after padding
# Dataset paths for SiamRCNN training
_C.INPUT.VQ_IMAGES_ROOT = "./images"
_C.INPUT.VQ_DATA_SPLITS_ROOT = "./vq_splits"


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
# Config for pre-trained weights for SiamRCNN
_C.MODEL.SIAMESE_PRETRAINED_CONFIG = (
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)


# ---------------------------------------------------------------------------- #
# Siamese Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_SIAMESE_HEAD = CfgNode()
_C.MODEL.ROI_SIAMESE_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.ROI_SIAMESE_HEAD.QUERY_FEATURE = "p3"
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_TYPE = "ROIAlignV2"
# Hidden size for siamese similarity head
_C.MODEL.ROI_SIAMESE_HEAD.HIDDEN_SIZE = 1024
# Projection layer hyperparameters
_C.MODEL.ROI_SIAMESE_HEAD.PROJECTOR_TYPE = "basic"
_C.MODEL.ROI_SIAMESE_HEAD.N_RESIDUAL_LAYERS = 1
# Hard negative mining for loss computation
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING = CfgNode()
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.ENABLE = False
# Number of hard negatives to mine from the set of all negatives
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.NUM_NEGATIVES = 16
# Loss type to use [ bce | kl_div | metric ] --- metric applies only for "dot"
_C.MODEL.ROI_SIAMESE_HEAD.LOSS_TYPE = "bce"
# Share the projection layers for siamese head?
_C.MODEL.ROI_SIAMESE_HEAD.SHARE_PROJECTION = False
# Compare layer type [ bilinear | dot ]
_C.MODEL.ROI_SIAMESE_HEAD.COMPARE_TYPE = "bilinear"
# Margin value for triplet-margin loss
_C.MODEL.ROI_SIAMESE_HEAD.TRIPLET_MARGIN = 0.25
# Enable cross batch negatives
_C.MODEL.ROI_SIAMESE_HEAD.USE_CROSS_BATCH_NEGATIVES = False


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()

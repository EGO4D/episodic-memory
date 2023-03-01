from dataclasses import dataclass, field
from typing import List

import yaml
from fvcore.common.config import CfgNode


"""
Add default config for STARK-ST Stage2.
"""
cfg = CfgNode()

# MODEL
cfg.MODEL = CfgNode()
cfg.MODEL.HEAD_TYPE = "CORNER"
cfg.MODEL.NLAYER_HEAD = 3
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = "sine"  # sine or learned
cfg.MODEL.PREDICT_MASK = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = CfgNode()
cfg.MODEL.BACKBONE.TYPE = "resnet50"  # resnet50, resnext101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ["layer3"]
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.DILATION = False
# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = CfgNode()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False

# TRAIN
# General TRAIN config
TRAIN = CfgNode()
TRAIN.TRAIN_CLS = False
TRAIN.LR = 0.0001
TRAIN.WEIGHT_DECAY = 0.0001
TRAIN.EPOCH = 50
TRAIN.LR_DROP_EPOCH = 40
TRAIN.BATCH_SIZE = 16
TRAIN.NUM_WORKER = 8
TRAIN.OPTIMIZER = "ADAMW"
TRAIN.BACKBONE_MULTIPLIER = 0.1
TRAIN.GIOU_WEIGHT = 2.0
TRAIN.L1_WEIGHT = 5.0
TRAIN.DEEP_SUPERVISION = False
TRAIN.FREEZE_BACKBONE_BN = True
TRAIN.FREEZE_LAYERS = ["conv1", "layer1"]
TRAIN.PRINT_INTERVAL = 50
TRAIN.VAL_EPOCH_INTERVAL = 20
TRAIN.GRAD_CLIP_NORM = 0.1
TRAIN.CHECKPOINT_PERIOD = 100
# TRAIN.SCHEDULER
TRAIN.SCHEDULER = CfgNode()
TRAIN.SCHEDULER.TYPE = "step"
TRAIN.SCHEDULER.DECAY_RATE = 0.1

cfg.TRAIN_STAGE_1 = TRAIN.clone()
cfg.TRAIN_STAGE_2 = TRAIN.clone()

cfg.TRAIN_STAGE_2.EPOCH = 10
cfg.TRAIN_STAGE_2.TRAIN_CLS = True

# DATA
cfg.DATA = CfgNode()
cfg.DATA.SAMPLER_MODE = "trident_pro"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = [200]
# DATA.TRAIN
cfg.DATA.TRAIN = CfgNode()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain", "COCO"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = CfgNode()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = CfgNode()
cfg.DATA.SEARCH.NUMBER = 1  # number of search frames for multiple frames training
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = CfgNode()
cfg.DATA.TEMPLATE.NUMBER = 2
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = CfgNode()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 50
cfg.TEST.UPDATE_INTERVALS = CfgNode()
cfg.TEST.UPDATE_INTERVALS.LASOT = [200]
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = [200]
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = [25]
cfg.TEST.UPDATE_INTERVALS.VOT20 = [10]
cfg.TEST.UPDATE_INTERVALS.VOT20LT = [200]
cfg.TEST.UPDATE_INTERVALS.EGO4D = [1]


@dataclass
class _DATA_TEMPLATE:
    NUMBER = 2
    SIZE = 128
    FACTOR = 2.0
    CENTER_JITTER = 0
    SCALE_JITTER = 0


@dataclass
class _DATA_SEARCH:
    NUMBER = 1  # number of search frames for multiple frames training
    SIZE = 320
    FACTOR = 5.0
    CENTER_JITTER = 4.5
    SCALE_JITTER = 0.5


@dataclass
class _DATA_VAL:
    DATASETS_NAME: List[str] = field(default_factory=lambda: ["GOT10K_votval"])
    DATASETS_RATIO: List[float] = field(default_factory=lambda: [1])
    SAMPLE_PER_EPOCH: int = 10000


@dataclass
class _DATA_TRAIN:
    DATASETS_NAME: List[str] = field(
        default_factory=lambda: ["LASOT", "GOT10K_vottrain", "COCO17"]
    )
    DATASETS_RATIO: List[float] = field(default_factory=lambda: [1, 1, 1])
    SAMPLE_PER_EPOCH: int = 60000


@dataclass
class _DATA:
    SAMPLER_MODE = "trident_pro"  # sampling methods
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    MAX_SAMPLE_INTERVAL: List[int] = field(default_factory=lambda: [200])
    COCO_DATA_DIR = "manifold://fair_vision_data/tree/"
    # COCO_DATA_DIR = "manifold://mscoco/tree/coco2017"
    LASOT_DATA_DIR: str = "manifold://fai4ar/tree/datasets/LaSOTBenchmark"
    TRACKINGNET_DATA_DIR: str = "manifold://tracking/tree/data/trackingnet"
    CACHED_TRACKINGNET_SEQUENCE_LIST_DIR: str = (
        "manifold://tracking/tree/data/trackingnet/cache"
    )
    GOT10K_DATA_DIR: str = "manifold://tracking/tree/data/got10k"
    CACHED_GOT10K_META_INFO_DIR: str = "manifold://tracking/tree/data/got10k/cache"
    EGO4DVQ_DATA_DIR: str = "manifold://tracking/tree/data/ego4d/vq_frames"
    EGO4DVQ_ANNOTATION_PATH: str = (
        "manifold://tracking/tree/ego4d/v1/annotations/vq_train.json"
    )
    TRAIN: _DATA_TRAIN = _DATA_TRAIN()
    VAL: _DATA_VAL = _DATA_VAL()
    SEARCH: _DATA_SEARCH = _DATA_SEARCH()
    TEMPLATE: _DATA_TEMPLATE = _DATA_TEMPLATE()
    DATA_FRACTION = None


@dataclass
class _SCHEDULER:
    TYPE: str = "step"
    DECAY_RATE: float = 0.1


@dataclass
class _TRAIN:
    TRAIN_CLS: bool = False
    LR: float = 0.0001
    WEIGHT_DECAY: float = 0.0001
    EPOCH: int = 500
    LR_DROP_EPOCH: int = 400
    BATCH_SIZE: int = 16
    NUM_WORKER: int = 8
    OPTIMIZER: str = "ADAMW"
    BACKBONE_MULTIPLIER: float = 0.1
    GIOU_WEIGHT: float = 2.0
    L1_WEIGHT: float = 5.0
    DEEP_SUPERVISION: bool = False
    FREEZE_BACKBONE_BN: bool = True
    FREEZE_LAYERS: List[str] = field(default_factory=lambda: ["conv1", "layer1"])
    PRINT_INTERVAL: int = 50
    VAL_EPOCH_INTERVAL: int = 20
    GRAD_CLIP_NORM: float = 0.1
    CHECKPOINT_PERIOD: int = 100
    RESUME: bool = False
    FINETUNE_CHECKPOINT: str = ""
    # TRAIN.SCHEDULER
    SCHEDULER: _SCHEDULER = _SCHEDULER()


@dataclass
class _MODEL_BACKBONE:
    TYPE = "resnet50"  # resnet50, resnext101_32x8d
    OUTPUT_LAYERS: List[str] = field(default_factory=lambda: ["layer3"])
    STRIDE = 16
    DILATION = False


@dataclass
class _MODEL_TRANSFORMER:
    NHEADS = 8
    DROPOUT = 0.1
    DIM_FEEDFORWARD = 2048
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    PRE_NORM = False
    DIVIDE_NORM = False


@dataclass
class _MODEL:
    HEAD_TYPE = "CORNER"
    NLAYER_HEAD = 3
    HIDDEN_DIM = 256
    NUM_OBJECT_QUERIES = 1
    POSITION_EMBEDDING = "sine"  # sine or learned
    PREDICT_MASK = False
    BACKBONE: _MODEL_BACKBONE = _MODEL_BACKBONE()
    TRANSFORMER: _MODEL_TRANSFORMER = _MODEL_TRANSFORMER()


@dataclass
class _UPDATE_INTERVALS:
    LASOT: List = field(default_factory=lambda: [200])
    GOT10K_TEST: List = field(default_factory=lambda: [200])
    TRACKINGNET: List = field(default_factory=lambda: [25])
    VOT20: List = field(default_factory=lambda: [10])
    VOT20LT: List = field(default_factory=lambda: [200])
    EGO4D: List = field(default_factory=lambda: [1])


@dataclass
class _TEST:
    TEMPLATE_FACTOR: float = 2.0
    TEMPLATE_SIZE: int = 128
    SEARCH_FACTOR: float = 5.0
    SEARCH_SIZE: int = 320
    EPOCH: int = 50
    TEST_BATCHSIZE: int = 8
    UPDATE_INTERVALS: _UPDATE_INTERVALS = _UPDATE_INTERVALS()


@dataclass
class STARKParams:
    TRAIN: _TRAIN = _TRAIN()
    DATA: _DATA = _DATA()
    TEST: _TEST = _TEST()
    MODEL: _MODEL = _MODEL()


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, CfgNode):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, CfgNode):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = CfgNode(yaml.safe_load(f))
        _update_config(cfg, exp_config)

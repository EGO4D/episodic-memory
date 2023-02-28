from detectron2_extensions.config import get_cfg as get_detectron_cfg
from vq2d.baselines import SiamPredictor


def test_model_loading(config_path, checkpoint_path):
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.set_new_allowed(True)
    detectron_cfg.merge_from_file(config_path)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    detectron_cfg.MODEL.WEIGHTS = checkpoint_path
    detectron_cfg.MODEL.DEVICE = f"cuda:0"
    detectron_cfg.INPUT.FORMAT = "RGB"
    predictor = SiamPredictor(detectron_cfg)
    print(predictor)


if __name__ == "__main__":
    test_model_loading(
        "pretrained_models/negative_frames_matter/config.yaml",
        "pretrained_models/negative_frames_matter/model.pth",
    )

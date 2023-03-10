# Download the model weight from https://drive.google.com/file/d/14vZmWxYSGJXZGxD5U1LthvvTR_eRzWCw/view?usp=share_link (EgoSTARK, trained using EgoTracks).
from tracking.config.stark_defaults import cfg as stark_cfg
from tracking.models.stark_tracker.stark_tracker import STARKTracker
import torch
import os
from tracking.utils.utils import opencv_loader, visualize_bbox
import matplotlib.pyplot as plt
import numpy as np


def main():
    egostark_cfg = stark_cfg.clone()
    local_rank = 0

    # EgoSTARK parameter
    egostark_cfg.MODEL.WEIGHTS = "/checkpoint/haotang/experiments/EgoTracks/STARKST_ep0001.pth.tar"
    egostark_cfg.TEST.SEARCH_SIZE = 640

    # Initilize model
    model = STARKTracker(
        egostark_cfg, device=torch.device(f"cuda:{local_rank}")
    )
    model.eval()

    # Read test images
    imgs_dir = "EgoTracks/test_images"
    images = [opencv_loader(os.path.join(imgs_dir, f)) for f in os.listdir(imgs_dir)]
    images = np.array(images)

    # Run inference
    frame_number = list(range(len(images)))
    target_bbox = [1100, 400, 400, 400]
    target_id = "tape"
    meta_data = {
        "target_bbox": target_bbox,
        "target_id": target_id,
        "frame_numbers": frame_number
    }
    pred_traj = model.inference(images, meta_data)

    # visualize and save
    bboxes = pred_traj[target_id]["bboxes"]
    bboxes = [b["bbox"] for b in bboxes]
    vis_imgs = visualize_bbox(images, bboxes)

    save_dir = "EgoTracks/test_viz"
    os.makedirs(save_dir, exist_ok=True)
    for i, im in enumerate(vis_imgs):
        plt.imsave(os.path.join(save_dir, f"{i}.jpg"), im)


if __name__ == "__main__":
    main()
from utils import *
from models.superpoint import SuperPoint
import argparse

import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--matterport_dataset_folder', type=str,
    default='/home/kvuong/dgx-projects/ego4d_data/Matterport/walterb18_testing',
    help='SuperGlue match threshold')
parser.add_argument(
    '--matterport_descriptors_outputdir', type=str,
    default='/home/kvuong/dgx-projects/ego4d_data/Matterport/walterb18_testing_descriptors',
    help='SuperGlue match threshold')

opt = parser.parse_args()

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }
}
MATTERPORT_DATASET_FOLDER = opt.matterport_dataset_folder
IMG_DESC_OUTPUT_DIR = opt.matterport_descriptors_outputdir

if not os.path.exists(IMG_DESC_OUTPUT_DIR):
    os.makedirs(IMG_DESC_OUTPUT_DIR)

K_mp = np.array([[700., 0., 960. - 0.5],
                 [0., 700., 540. - 0.5],
                 [0., 0., 1.]])

superpoint = SuperPoint(config.get('superpoint', {})).cuda()
superpoint = superpoint.eval()

matterport_dataset = MatterportDataset(dataset_folder=MATTERPORT_DATASET_FOLDER)
data_loader = DataLoader(dataset=matterport_dataset, num_workers=4, batch_size=12, shuffle=False, pin_memory=True)

img_idx = 0
with torch.no_grad():
    for idx, images in enumerate(tqdm(data_loader)):
        # print(images['image_index'])
        images['image'] = images['image'].cuda()
        output = superpoint(images)
        for j in range(images['image'].shape[0]):
            mp_idx = images['image_index'][j]
            image_descriptors = output['descriptors'][j].detach().cpu().numpy()  # DxN = 256xNum_KPs
            keypoints = output['keypoints'][j].detach().cpu().numpy()
            scores = output['scores'][j].detach().cpu().numpy()

            d1 = Image.open(os.path.join(MATTERPORT_DATASET_FOLDER, 'depth/depth_%06d.png' % mp_idx))
            d1 = np.asarray(d1).astype(np.float32) / 1000.0
            T1 = np.loadtxt(os.path.join(MATTERPORT_DATASET_FOLDER, 'pose/pose_%06d.txt' % mp_idx))

            pts3d_all = None
            f3d = []
            flag_array = np.zeros((keypoints.shape[0]))
            for i in range(keypoints.shape[0]):
                pt2d = keypoints[i] * np.array([3.0, 2.25]) # SuperPoint operates on 480x640, so must scale to 1280x1920
                u, v = pt2d[0], pt2d[1]
                z = d1[int(v), int(u)]
                if z > 0.0:
                    flag_array[i] = 1.0
                xyz_curr = convert_2d_to_3d(u, v, z, K_mp)
                f3d.append(xyz_curr)
            f3d = (T1[:3, :3] @ np.array(f3d).transpose() + T1[:3, 3:]).transpose()
            pts3d_all = np.diag(flag_array) @ f3d

            # the minimal number of points accepted by solvePnP is 4:
            f3d = np.expand_dims(pts3d_all.astype(np.float32), axis=1)

            # Write the matches to disk.
            out_matches = {'keypoints': keypoints,
                           'scores': scores,
                           'descriptors': image_descriptors,
                           'XYZ': f3d}
            np.savez(os.path.join(IMG_DESC_OUTPUT_DIR, 'image_%06d_descriptors.npz' % img_idx), **out_matches)
            img_idx += 1

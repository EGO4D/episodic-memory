import torch
import fnmatch
import argparse
import numpy as np
from torch.utils.data.dataset import Dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from utils import WritePosesToPly

def PnP(x1s, f3ds, x2s, m1_ids, K1, K2, thres=1e-2):
    f2d = [] # keep only feature points with depth in the current frame
    f3d_new = []
    for k in range(len(f3ds)):
        x2 = np.array(x2s[k])
        f3d = np.array(f3ds[k])
        for i in range(f3d.shape[0]):
            if np.linalg.norm(f3d[i]) > 1e-2:
                f2d.append(x2[i, :])
                f3d_new.append(f3d[i, 0])

    f2d = np.array(f2d)
    f3d = np.array(f3d_new)

    if f3d.ndim == 2:
        f3d = np.expand_dims(f3d.astype(np.float32), axis=1)

    if f2d.ndim == 2:
        f2d = np.expand_dims(f2d.astype(np.float32), axis=1)

    if (f3d.shape[0]<4) or (f2d.shape[0]<4):
        return 0, None, None, None

    ret = cv2.solvePnPRansac(f3d,
                             f2d,
                             K2,
                             distCoeffs=None,
                             flags=cv2.SOLVEPNP_EPNP)
    success = ret[0]
    rotation_vector = ret[1]
    translation_vector = ret[2]

    f_2d = np.linalg.inv(K2) @ np.concatenate((f2d[:, 0],
                                               np.ones((f2d.shape[0], 1))), axis=1).T

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = translation_vector.reshape(3)
    proj = rotation_mat @ f3d[:, 0].T + translation_vector.reshape(3, -1)
    proj = proj[:2] / proj[2:]
    reproj_error = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
    reproj_inliers = reproj_error < thres
    reproj_inliers = reproj_inliers.reshape(-1)

    if success==0 or reproj_inliers.sum() < 10:
        return 0, None, None, None
    else:
        ret = cv2.solvePnP(f3d[reproj_inliers].reshape(reproj_inliers.sum(), 1, 3),
                           f2d[reproj_inliers].reshape(reproj_inliers.sum(), 1, 2),
                           K2,
                           distCoeffs=None,
                           flags=cv2.SOLVEPNP_ITERATIVE)
        success = ret[0]
        rotation_vector = ret[1]
        translation_vector = ret[2]

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = translation_vector.reshape(3)

        Caz_T_Wmp = np.eye(4)
        Caz_T_Wmp[:3, :3] = rotation_mat
        Caz_T_Wmp[:3, 3] = translation_vector

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = translation_vector.reshape(3)
        proj = rotation_mat @ f3d[:, 0].T + translation_vector.reshape(3, -1)
        proj = proj[:2] / proj[2:]
        reproj_error_refined = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
        reproj_error_refined = reproj_error_refined < thres
        reproj_error_refined = reproj_error_refined.reshape(-1)

        if reproj_error_refined.sum() < 0.5 * reproj_inliers.sum():
            return 0, None, None, None
        else:
            return success, Caz_T_Wmp, f2d[reproj_error_refined, 0], f3d[reproj_error_refined, 0]

class PosePnP_points_accumulation(Dataset):
    def __init__(self, match_database='', img_desc_folder='', image_list=None):
        super(PosePnP_points_accumulation, self).__init__()
        self.img_desc_folder = img_desc_folder
        self.match_database = match_database
        self.num_images = len(image_list)
        self.P = np.zeros((self.num_images, 3, 4))
        self.good_pose_pnp = np.zeros(self.num_images, dtype=bool)
        self.original_image_id_list = image_list

    def __getitem__(self, index):
        azure_img_idx = self.original_image_id_list[index]
        matches_file_list = fnmatch.filIter(os.listdir(self.match_database), 
                                           'color_%07d_*_matches.npz' % azure_img_idx)

        best_inlier = -1
        best_solution = None
        output = {'img_idx': torch.tensor(index, dtype=torch.int),
                  'is_good_pose': torch.tensor([False]),
                  'solution': torch.zeros((3, 4), dtype=torch.double)}

        x2_all = []
        f3d_all = []
        total_inlier = 0
        for file_idx in range(len(matches_file_list)):
            # 1. Input an query RGB from Kinect Azure
            matterport_img_idx = int(matches_file_list[file_idx][20:26])

            matches_data = np.load(os.path.join(self.match_database, matches_file_list[file_idx]),'r')

            image_descriptor = np.load(os.path.join(self.img_desc_folder, 'image_%06d_descriptors.npz' % matterport_img_idx),'r')

            _x1 = []
            _f3d = []
            _x2 = []
            good_matches = 0
            
            for i in range(matches_data['keypoints0'].shape[0]):
                if matches_data['matches'][i] >= 0 and matches_data['match_confidence'][i] > 0.1:
                    _x2.append(matches_data['keypoints0'][i] * np.array(s_ego))
                    _x1.append(matches_data['keypoints1'][matches_data['matches'][i]] * np.array(s_mp))
                    _f3d.append(image_descriptor['XYZ'][matches_data['matches'][i]])
                    good_matches += 1
             
            if good_matches > 20:
                success, T, f2d_inlier, f3d_inlier = PnP([_x1], [_f3d], [_x2], [matterport_img_idx],
                                                         K_mp, K_ego, thres=3e-2)

                if success and f2d_inlier.shape[0] >= 10:
                    total_inlier += f2d_inlier.shape[0]
                    x2_all.append(f2d_inlier)
                    f3d_all.append(f3d_inlier)

        if total_inlier >= 30:
            f3d_all = np.concatenate(f3d_all, axis=0)
            f3d_all = np.expand_dims(f3d_all, axis=1)
            x2_all = np.concatenate(x2_all, axis=0)
            success, T, f2d_inlier, f3d_inlier = PnP(None, [f3d_all], [x2_all], None, K_mp, K_ego, thres=3e-2)
            if success and f2d_inlier.shape[0] > 20:
                ## VISUALIZATION
                # uv1 = np.concatenate((f2d_inlier,
                #                       np.ones((f2d_inlier.shape[0], 1))), axis=1)
                # azure_im = cv2.imread(os.path.join(EGOLOC_FOLDER, 'color/color_%07d.jpg' % azure_img_idx))
                # VisualizeReprojectionError(T[:3],
                #                            f3d_inlier,
                #                            uv1 @ np.linalg.inv(K_azure).T,
                #                            Im=azure_im, K=K_azure)
                # self.P[index] = copy.deepcopy(T[:3])
                # self.good_pose_pnp[index] = copy.deepcopy(True)

                output = {'img_idx': torch.tensor(index, dtype=torch.int),
                          'is_good_pose': torch.tensor([True]), 'solution': torch.tensor(T[:3])}

        return output

    def __len__(self):
        return self.num_images



parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--ego_dataset_folder', type=str, default='',
    help='SuperGlue match threshold')
parser.add_argument(
    '--matterport_descriptors_folder', type=str,
    help='Matterport descriptor folder')
parser.add_argument(
    '--output_dir', type=str,
    help='SuperGlue match threshold')

opt = parser.parse_args()

EGO_DATASET_FOLDER = opt.ego_dataset_folder
MATCH_DATABASE = os.path.join(EGO_DATASET_FOLDER, 'superglue_match_results')
IMAGE_DESC_FOLDER = opt.matterport_descriptors_folder
OUTPUT_DIR = opt.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

K_mp = np.array([[700., 0., 960. - 0.5],
                 [0., 700., 540. - 0.5],
                 [0., 0., 1.]])
s_mp = [(K_mp[0,2]+0.5)*2/640, (K_mp[1,2]+0.5)*2/480]

K_ego = np.loadtxt('%s/intrinsics.txt' % opt.ego_dataset_folder)
s_ego = [(K_ego[0,2]+0.5)*2/640, (K_ego[1,2]+0.5)*2/480]

original_image_id_list = np.arange(0, len(fnmatch.filter(os.listdir(EGO_DATASET_FOLDER + '/color/'), '*.jpg')), step=1)
num_images = original_image_id_list.shape[0]
P = np.zeros((num_images, 3, 4))
good_pose_pnp = np.zeros(num_images, dtype=bool)
batch_size = 8

ego_dataset = PosePnP_points_accumulation(match_database=MATCH_DATABASE,
                                         img_desc_folder=IMAGE_DESC_FOLDER,
                                         image_list=original_image_id_list)
data_loader = DataLoader(dataset=ego_dataset,
                         num_workers=8, batch_size=batch_size,
                         shuffle=False,
                         pin_memory=True)

for idx, output_batch in enumerate(tqdm(data_loader)):
    for ii in range(output_batch['is_good_pose'].shape[0]):
        if output_batch['is_good_pose'][ii]:
            P[int(output_batch['img_idx'][ii].item())] = output_batch['solution'][ii].numpy()
            good_pose_pnp[int(output_batch['img_idx'][ii].item())] = True

    print('good pose found by pnp / total poses: ', np.sum(good_pose_pnp), '/', good_pose_pnp.shape[0])

np.save('%s/camera_poses_pnp.npy' % OUTPUT_DIR, P)
np.save('%s/good_pose_pnp.npy' % OUTPUT_DIR, good_pose_pnp)
print(good_pose_pnp.shape)
if np.sum(good_pose_pnp) > 0:
    WritePosesToPly(P[good_pose_pnp], '%s/cameras_pnp.ply' % OUTPUT_DIR)
else:
    print('No good poses found')

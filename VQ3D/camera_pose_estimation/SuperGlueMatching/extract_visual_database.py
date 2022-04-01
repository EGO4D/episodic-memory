import torch
import cv2
import fnmatch

from utils import *
from models.superpoint import SuperPoint
import argparse
import random
import numpy as np
import matplotlib.cm as cm

from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader

from models.utils import read_image
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import pickle


def convert_2d_to_3d(u, v, z, K):
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    return x, y, z


class MatterportDataset(Dataset):
    def __init__(self, dataset_name='walterlib', resize=[640, 480], root=''):
        super(MatterportDataset, self).__init__()
        self.root_dir = root
        self.data_path = os.path.join(self.root_dir, dataset_name, 'color')
        self.data_info = sorted(os.listdir(self.data_path))
        # self.data_info = self.data_info[:10]
        self.data_len = len(self.data_info)
        self.resize = resize

    def __getitem__(self, index):
        color_info = os.path.join(self.data_path, self.data_info[index])
        # print(color_info)
        _, gray_tensor, _ = read_image(color_info, 'cpu', resize=self.resize, rotation=0, resize_float=False)
        gray_tensor = gray_tensor.reshape(1, 480, 640)

        output = {'image': gray_tensor, 'image_index': int(color_info[-10:-4])}

        return output

    def __len__(self):
        return self.data_len


class AzureKinect(Dataset):
    def __init__(self, dataset_name='walter_basement_03', resize=[640, 480], root='',
                 start_idx=0, end_idx=10000, skip_every_n_image=1):
        super(AzureKinect, self).__init__()
        self.root_dir = root
        self.data_path = os.path.join(self.root_dir, dataset_name, 'color')
        self.data_info = fnmatch.filter(os.listdir(self.data_path), '*.jpg')
        self.data_info = self.data_info[start_idx:end_idx:skip_every_n_image]
        self.start_idx = start_idx
        self.data_len = len(self.data_info)
        self.resize = resize

    def __getitem__(self, index):
        color_info = os.path.join(self.data_path, 'color_%07d.jpg' % (self.start_idx + index))
        # print(color_info)
        _, gray_tensor, _ = read_image(color_info, 'cpu', resize=self.resize, rotation=0, resize_float=False)
        gray_tensor = gray_tensor.reshape(1, 480, 640)

        output = {'image': gray_tensor, 'image_index': int(color_info[-11:-4])}

        return output

    def __len__(self):
        return self.data_len


def PnP(x1s, f3ds, x2s, m1_ids, K1, K2, DATABASE_IMAGE_FOLDER):
    # # extract 3d pts
    # pts3d_all = None
    # f2d = []  # keep only feature points with depth in the current frame
    # for k in range(len(x1s)):
    #     d1 = Image.open(os.path.join(DATABASE_IMAGE_FOLDER, 'depth/depth_%06d.png' % m1_ids[k]))
    #     d1 = np.asarray(d1).astype(np.float32) / 1000.0
    #     x1 = np.array(x1s[k])
    #     x2 = np.array(x2s[k])
    #     T1 = np.loadtxt(os.path.join(DATABASE_IMAGE_FOLDER, 'pose/pose_%06d.txt' % m1_ids[k]))
    #
    #     f3d = []
    #     for i, pt2d in enumerate(x1):
    #         u, v = pt2d[0], pt2d[1]
    #         z = d1[int(v), int(u)]
    #         if z > 0:
    #             xyz_curr = convert_2d_to_3d(u, v, z, K1)
    #             f3d.append(xyz_curr)
    #             f2d.append(x2[i, :])
    #         # else:
    #         #     print(convert_2d_to_3d(u, v, z, K1), f3d_loaded[i, 0])
    #
    #     f3d = (T1[:3, :3] @ np.array(f3d).transpose() + T1[:3, 3:]).transpose()
    #     if pts3d_all is None:
    #         pts3d_all = f3d
    #     else:
    #         pts3d_all = np.concatenate((pts3d_all, f3d), axis=0)
    #
    # # the minimal number of points accepted by solvePnP is 4:
    # f3d = np.expand_dims(pts3d_all.astype(np.float32), axis=1)
    # print(f3d.shape)

    f2d = [] # keep only feature points with depth in the current frame
    f3d_new = []
    for k in range(len(f3ds)):
        x2 = np.array(x2s[k])
        f3d = np.array(f3ds[k])
        for i in range(f3d.shape[0]):
            if f3d[i, 0, 0] != 0.0 or f3d[i, 0, 1] != 0.0 or f3d[i, 0, 2] != 0.0:
                f2d.append(x2[i, :])
                f3d_new.append(f3d[i, 0])

    # the minimal number of points accepted by solvePnP is 4:
    f3d = np.expand_dims(np.array(f3d_new).astype(np.float32), axis=1)

    f2d = np.expand_dims(
        np.array(f2d).astype(np.float32), axis=1)

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
    reproj_inliers = reproj_error < 1e-2
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
        reproj_error_refined = reproj_error_refined < 1e-2
        reproj_error_refined = reproj_error_refined.reshape(-1)

        if reproj_error_refined.sum() < 0.8 * reproj_inliers.sum():
            return 0, None, None, None
        else:
            return success, Caz_T_Wmp, f2d[reproj_error_refined, 0], f3d[reproj_error_refined, 0]


K_mp = np.array([[700., 0., 960. - 0.5],
                 [0., 700., 540. - 0.5],
                 [0., 0., 1.]])

K_azure = np.array([[913., 0., 960. - 0.5],
                    [0., 913., 540. - 0.5],
                    [0., 0., 1.]])

class AzureKinectPosePnP(Dataset):
    def __init__(self, match_database='', database_image_folder='', image_list=None):
        super(AzureKinectPosePnP, self).__init__()
        self.database_image_folder = database_image_folder
        self.match_database = match_database
        self.num_images = len(image_list)
        self.P = np.zeros((self.num_images, 3, 4))
        self.good_pose_pnp = np.zeros(self.num_images, dtype=bool)
        self.original_image_id_list = image_list

    def __getitem__(self, index):
        azure_img_idx = self.original_image_id_list[index]
        # matches_file_list = fnmatch.filter(os.listdir(self.match_database), '*_color_%07d_matches.npz' % azure_img_idx)
        matches_file_list = fnmatch.filter(os.listdir(self.match_database), 'color_%07d_*_matches.npz' % azure_img_idx)

        best_inlier = -1
        best_solution = None
        output = {'img_idx': torch.tensor(index, dtype=torch.int),
                  'is_good_pose': torch.tensor([False]),
                  'solution': torch.zeros((3, 4), dtype=torch.double)}
        for file_idx in range(len(matches_file_list)):
            # 1. Input an query RGB from Kinect Azure
            # matterport_img_idx = int(matches_file_list[file_idx][6:12])
            matterport_img_idx = int(matches_file_list[file_idx][20:26])
            matches_data = np.load(os.path.join(self.match_database, matches_file_list[file_idx]))

            IMAGE_DESC_FOLDER = os.path.join(ROOT_DIR, DATASET_NAME, 'image_descriptors_ours')
            image_descriptor = np.load(os.path.join(IMAGE_DESC_FOLDER, 'image_%06d_descriptors.npz' % matterport_img_idx))

            _x1 = []
            _f3d = []
            _x2 = []
            good_matches = 0

            for i in range(matches_data['keypoints0'].shape[0]):
                if matches_data['matches'][i] >= 0 and matches_data['match_confidence'][i] > 0.2:
                    _x2.append(matches_data['keypoints0'][i] * np.array([3.0, 2.25]))
                    _x1.append(matches_data['keypoints1'][matches_data['matches'][i]] * np.array([3.0, 2.25]))
                    _f3d.append(image_descriptor['XYZ'][matches_data['matches'][i]])
                    good_matches += 1

            if good_matches > 30:
                success, T, f2d_inlier, f3d_inlier = PnP([_x1], [_f3d], [_x2], [matterport_img_idx],
                                                         K_mp, K_azure, self.database_image_folder)

                if success and f2d_inlier.shape[0] >= 20:
                    if f2d_inlier.shape[0] > best_inlier:
                        best_solution = T, f2d_inlier, f3d_inlier
                        best_inlier = f2d_inlier.shape[0]

        if best_solution is not None:
            T, f2d_inlier, f3d_inlier = best_solution
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
    '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs')
parser.add_argument(
    '--input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optionally,'
         'the visualization images are written')

parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
         'resize to the exact dimensions, if one number, resize the max '
         'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

opt = parser.parse_args()
print(opt)

config = {
    'superpoint': {
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints
    },
    'superglue': {
        'weights': opt.superglue,
        'sinkhorn_iterations': opt.sinkhorn_iterations,
        'match_threshold': opt.match_threshold,
    }
}

ROOT_DIR = '/home/kvuong/dgx-projects/ego4d_data/Matterport'
DATASET_NAME = 'walterb18'

if not os.path.exists(os.path.join(ROOT_DIR, DATASET_NAME, 'features')):
    os.makedirs(os.path.join(ROOT_DIR, DATASET_NAME, 'features'))


superpoint = SuperPoint(config.get('superpoint', {})).cuda()
superpoint = superpoint.eval()

matterport_dataset = MatterportDataset(root=ROOT_DIR, dataset_name=DATASET_NAME)
data_loader = DataLoader(dataset=matterport_dataset,
                            num_workers=4, batch_size=12,
                            shuffle=False,
                            pin_memory=True)

K_mp = np.array([[700., 0., 960. - 0.5],
                 [0., 700., 540. - 0.5],
                 [0., 0., 1.]])

# ## Extract superpoint for each MP image (theirs)
if not os.path.exists(os.path.join(ROOT_DIR, DATASET_NAME, 'image_descriptors_ours')):
    os.makedirs(os.path.join(ROOT_DIR, DATASET_NAME, 'image_descriptors_ours'))


# with torch.no_grad():
#     for idx, images in enumerate(data_loader):
#         print(images['image_index'])
#         images['image'] = images['image'].cuda() # 480x640
#         output = superpoint(images)
#         for j in range(images['image'].shape[0]):
#             mp_idx = images['image_index'][j]
#             image_descriptors = output['descriptors'][j].detach().cpu().numpy()  # DxN = 256xNum_KPs
#             keypoints = output['keypoints'][j].detach().cpu().numpy()
#             scores = output['scores'][j].detach().cpu().numpy()
#
#             d1 = Image.open(os.path.join(ROOT_DIR, DATASET_NAME, 'depth/depth_%06d.png' % mp_idx))
#             d1 = np.asarray(d1).astype(np.float32) / 1000.0
#             T1 = np.loadtxt(os.path.join(ROOT_DIR, DATASET_NAME, 'pose/pose_%06d.txt' % mp_idx))
#
#             pts3d_all = None
#             f3d = []
#             flag_array = np.zeros((keypoints.shape[0]))
#             for i in range(keypoints.shape[0]):
#                 pt2d = keypoints[i] * np.array([3.0, 2.25])
#                 u, v = pt2d[0], pt2d[1]
#                 z = d1[int(v), int(u)]
#                 if z > 0.0:
#                     flag_array[i] = 1.0
#                 xyz_curr = convert_2d_to_3d(u, v, z, K_mp)
#                 f3d.append(xyz_curr)
#             f3d = (T1[:3, :3] @ np.array(f3d).transpose() + T1[:3, 3:]).transpose()
#             pts3d_all = np.diag(flag_array) @ f3d
#
#             # the minimal number of points accepted by solvePnP is 4:
#             f3d = np.expand_dims(pts3d_all.astype(np.float32), axis=1)
#
#             # Write the matches to disk.
#             out_matches = {'keypoints': keypoints,
#                            'scores': scores,
#                            'descriptors': image_descriptors,
#                            'XYZ': f3d}
#             np.savez(os.path.join(ROOT_DIR, DATASET_NAME, 'image_descriptors_ours', 'image_%06d_descriptors.npz' % mp_idx), **out_matches)

'''

# # ### Ours
# ############# STEP 0: CONSTRUCT DESCRIPTOR CENTERS
total_removal = 0
all_descriptors = None
IMAGE_DESC_FOLDER = os.path.join(ROOT_DIR, DATASET_NAME, 'image_descriptors_ours')

# with torch.no_grad():
#     matterport_filelist = sorted(fnmatch.filter(os.listdir(IMAGE_DESC_FOLDER), 'image_*_descriptors.npz'))
#
#     for filename in matterport_filelist:
#         image_descriptors = np.load(os.path.join(IMAGE_DESC_FOLDER, filename))['descriptors']
#         _, N = image_descriptors.shape
#         if N < 300:
#             total_removal += 1
#             print('removed ', total_removal, ' images')
#             continue
#         if all_descriptors is None:
#             all_descriptors = np.transpose(image_descriptors, (1, 0))
#         else:
#             all_descriptors = np.concatenate((all_descriptors,
#                                               np.transpose(image_descriptors, (1, 0))), axis=0)
#             print('all_descriptors shape: ', all_descriptors.shape)
#
#
# kmeans = KMeans(n_clusters=64, random_state=0, init='k-means++', max_iter=5000, verbose=1).fit(all_descriptors)  # data
# np.save(os.path.join(ROOT_DIR, DATASET_NAME, 'data_descriptors_centers_ours.npy'), kmeans.cluster_centers_)


# # ############# STEP 1: BUILD IMAGE DESCRIPTOR TREE
descriptors_cluster_centers = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'data_descriptors_centers_ours.npy'))
descriptors_cluster_centers = torch.tensor(descriptors_cluster_centers, device='cuda')  # KxD
K, D = descriptors_cluster_centers.shape

# image_indices = []
# vlad_image_descriptors = []
#
# with torch.no_grad():
#     matterport_filelist = sorted(fnmatch.filter(os.listdir(IMAGE_DESC_FOLDER), 'image_*_descriptors.npz'))
#
#     for filename in matterport_filelist:
#         print(filename)
#         image_descriptors = np.load(os.path.join(IMAGE_DESC_FOLDER, filename))['descriptors']
#         image_descriptors = torch.tensor(image_descriptors, device='cuda')
#         _, N = image_descriptors.shape
#         if N < 250:
#             continue
#
#         # compute vlad image descriptor
#         assignment_matrix = descriptors_cluster_centers @ image_descriptors  # KxN
#         v = torch.max(assignment_matrix, dim=0)
#         assignment_mask = assignment_matrix == v.values.reshape(1, N).repeat((K, 1))  # KxN
#         assignment_mask = assignment_mask.reshape(1, K, N).repeat((D, 1, 1))
#         assignment_mask = assignment_mask.float()
#
#         image_descriptors = image_descriptors.reshape(D, 1, N).repeat((1, K, 1))  # DxKxN
#         residual = image_descriptors - torch.transpose(descriptors_cluster_centers, 0, 1).reshape(D, K, 1)  # DxKxN
#         masked_residual = torch.sum(assignment_mask * residual, dim=2)  # DxK
#
#         masked_residual = torch.nn.functional.normalize(masked_residual, p=2, dim=0)
#         vlad_image_descriptor = torch.nn.functional.normalize(masked_residual.reshape(1, -1), p=2, dim=1)
#
#         image_indices.append(int(filename[6:12]))
#         vlad_image_descriptors.append(vlad_image_descriptor.detach().cpu().numpy())
#
# vlad_image_descriptors = np.concatenate(vlad_image_descriptors, axis=0)
# np.save(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_ours.npy'), vlad_image_descriptors)
# np.save(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_indices_ours.npy'), np.asarray(image_indices))
# print('vlad_image_descriptors: ', vlad_image_descriptors.shape)

vlad_image_descriptors_ours = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_ours.npy'))
image_indices_ours = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_indices_ours.npy'))
vlad_image_descriptors = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors.npy'))
image_indices = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_indices.npy'))
#
## Cross-checking for correctness
print(vlad_image_descriptors_ours - vlad_image_descriptors)
print(np.sum(image_indices_ours - image_indices))
# print(np.linalg.norm(vlad_image_descriptors_ours[0] - vlad_image_descriptors[-14]))
#
# # for chosen_idx in range(image_indices_ours.shape[0]):
# #     for i in range(image_indices.shape[0]):
# #         if image_indices[i] == image_indices_ours[chosen_idx]:
# #             print(np.linalg.norm(vlad_image_descriptors_ours[chosen_idx] - vlad_image_descriptors[i]))
# ### Cross-checking for correctness


# ######### STEP 2: QUERY IMAGEG FROM DATABASE
kdt = KDTree(vlad_image_descriptors, leaf_size=30, metric='euclidean')

AZURE_ROOT_DIR = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure'
# # AZURE_DATASET_NAME = 'tien_walter_02'
# # AZURE_DATASET_NAME = 'walter_basement_03'
AZURE_DATASET_NAME = 'walterb18'

if not os.path.exists(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match_ours')):
    os.makedirs(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match_ours'))

ego_dataset = AzureKinect(root=AZURE_ROOT_DIR, dataset_name=AZURE_DATASET_NAME, skip_every_n_image=1,
                          start_idx=1140, end_idx=9300)
data_loader = DataLoader(dataset=ego_dataset,
                            num_workers=4, batch_size=12,
                            shuffle=False,
                            pin_memory=True)

match_pair = {'matterport': [], 'ego': []}
with torch.no_grad():
    for idx, images in enumerate(data_loader):
        print(images['image_index'])
        images['image'] = images['image'].cuda()
        output = superpoint(images)
        for j in range(images['image'].shape[0]):
            image_descriptors = output['descriptors'][j]  # DxN
            _, N = image_descriptors.shape

            # compute vlad image descriptor
            assignment_matrix = descriptors_cluster_centers @ image_descriptors  # KxN
            v = torch.max(assignment_matrix, dim=0)
            assignment_mask = assignment_matrix == v.values.reshape(1, N).repeat((K, 1))  # KxN
            assignment_mask = assignment_mask.reshape(1, K, N).repeat((D, 1, 1))
            assignment_mask = assignment_mask.float()

            image_descriptors = image_descriptors.reshape(D, 1, N).repeat((1, K, 1))  # DxKxN
            residual = image_descriptors - torch.transpose(descriptors_cluster_centers, 0, 1).reshape(D, K, 1)  # DxKxN
            masked_residual = torch.sum(assignment_mask * residual, dim=2)  # DxK

            masked_residual = torch.nn.functional.normalize(masked_residual, p=2, dim=0)
            vlad_image_descriptor = torch.nn.functional.normalize(masked_residual.reshape(1, -1), p=2, dim=1)

            # Match and save
            ret_query = kdt.query(vlad_image_descriptor.detach().cpu().numpy().reshape(1, -1), k=5, return_distance=False)[0]

            # # visualization
            # query_imgs = [cv2.imread(
            #     os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'color', 'color_%07d.jpg' % images['image_index'][j]))]
            # for i in range(5):
            #     query_imgs.append(cv2.imread(
            #         os.path.join(ROOT_DIR, DATASET_NAME, 'scolor', 'color_%06d.png' % image_indices[ret_query[i]])))
            # query_imgs = np.concatenate(query_imgs, axis=1)
            # cv2.imwrite('query_viz/%d.png' % images['image_index'][j], query_imgs)

            # save image matches
            # np.savetxt(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match', 'queries_%06d.txt' % images['image_index'][j]),
            #            np.array(image_indices[ret_query]), fmt='%d')
            for qids in range(len(ret_query)):
                match_pair['matterport'].append(
                    'Matterport/%s/color/color_%06d.jpg' % (DATASET_NAME, image_indices_ours[ret_query[qids]]))
                match_pair['ego'].append(
                    'KinectAzure/%s/color/color_%07d.jpg' % (AZURE_DATASET_NAME, images['image_index'][j]))

with open(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match_ours', 'queries_ours.pkl'), 'wb') as f:
    pickle.dump(match_pair, f)


### Cross-checking for correctness
queries_ours = pickle.load(open(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match_ours', 'queries_ours.pkl'), 'rb'))
queries = pickle.load(open(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match', 'queries.pkl'), 'rb'))
for i in range(len(queries['ego'])):
    if queries['ego'][i] != queries_ours['ego'][i]:
        print(queries['ego'][i], queries_ours['ego'][i])
    if queries['matterport'][i] != queries_ours['matterport'][i]:
        print(queries['matterport'][i], queries_ours['matterport'][i])
### Cross-checking for correctness


# Run PnP

# ### First, let's check if the matches are correct
# MATCH_DATABASE = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure/walterb18/superglue_match_results_redo/'
# MATCH_DATABASE_OURS = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure/walterb18/superglue_match_results_ours/'
# for azure_img_idx in range(1140, 1221):
#     matches_file_list = fnmatch.filter(os.listdir(MATCH_DATABASE), 'color_%07d_*_matches.npz' % azure_img_idx)
#     matches_file_list_ours = fnmatch.filter(os.listdir(MATCH_DATABASE_OURS), 'color_%07d_*_matches.npz' % azure_img_idx)
#     for file_idx in range(len(matches_file_list)):
#         print(file_idx)
#         matches_data = np.load(os.path.join(MATCH_DATABASE, matches_file_list[file_idx]))
#         matches_data_ours = np.load(os.path.join(MATCH_DATABASE_OURS, matches_file_list_ours[file_idx]))
#         print(np.linalg.norm(matches_data['keypoints0'] - matches_data_ours['keypoints0']))
#         print(np.linalg.norm(matches_data['keypoints1'] - matches_data_ours['keypoints1']))
#         print(np.linalg.norm(matches_data['matches'] - matches_data_ours['matches']))
# ### First, let's check if the matches are correct

'''

## Load all initial poses
EGOLOC_FOLDER = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure/walterb18/'
MATCH_DATABASE = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure/walterb18/superglue_match_results_ours/'
DATABASE_IMAGE_FOLDER = '/home/kvuong/dgx-projects/ego4d_data/Matterport/walterb18/'
OUTPUT_MATCH = '/home/kvuong/dgx-projects/ego4d_data/Matterport/walterb18/best_match_ours/'
AZURE_ROOT_DIR = '/home/kvuong/dgx-projects/ego4d_data/KinectAzure/'
AZURE_DATASET_NAME = 'walterb18'

OUTPUT_DIR = os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'ba_results_ours')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

original_image_id_list = np.arange(1140, 9300, step=1)
num_images = original_image_id_list.shape[0]
P = np.zeros((num_images, 3, 4))
good_pose_pnp = np.zeros(num_images, dtype=bool)
batch_size = 8


ego_dataset = AzureKinectPosePnP(match_database=MATCH_DATABASE,
                                 database_image_folder=DATABASE_IMAGE_FOLDER,
                                 image_list=original_image_id_list)
data_loader = DataLoader(dataset=ego_dataset,
                         num_workers=8, batch_size=batch_size,
                         shuffle=False,
                         pin_memory=True)

for idx, output_batch in enumerate(data_loader):
    print(output_batch['img_idx'])
    for ii in range(output_batch['is_good_pose'].shape[0]):
        if output_batch['is_good_pose'][ii]:
            P[int(output_batch['img_idx'][ii].item())] = output_batch['solution'][ii].numpy()
            good_pose_pnp[int(output_batch['img_idx'][ii].item())] = True

print('good pose found by pnp: ', np.sum(good_pose_pnp))
np.save('%s/camera_poses_pnp.npy' % OUTPUT_DIR, P)
np.save('%s/good_pose_pnp.npy' % OUTPUT_DIR, good_pose_pnp)
WritePosesToPly(P[good_pose_pnp], '%s/cameras_pnp.ply' % OUTPUT_DIR)

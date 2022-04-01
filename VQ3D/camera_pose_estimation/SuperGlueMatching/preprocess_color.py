import cv2
import json
import numpy as np
import os

dataset_path = "/home/kvuong/dgx-projects/ego4d_data/ego_videos/tong_ego_005"
color_path = os.path.join(dataset_path, "color_raw")

rgb_intrinsics_path = os.path.join(dataset_path, 'rgb_intrinsics.json')

# First read the camera intrinsics.
with open(rgb_intrinsics_path, 'r') as fp:
    intrinsics = json.load(fp)

downsample_factor = 2
k1 = intrinsics['k1']
k2 = intrinsics['k2']
k3 = intrinsics['k3']
k4 = intrinsics['k4']
k5 = intrinsics['k5']
k6 = intrinsics['k6']
p1 = intrinsics['p1']
p2 = intrinsics['p2']

fx = intrinsics['fx']
fy = intrinsics['fy']
cx = intrinsics['cx']
cy = intrinsics['cy']

if downsample_factor != 1:
    fx = fx / downsample_factor
    fy = fy / downsample_factor
    cx = cx / downsample_factor
    cy = cy / downsample_factor
    print('Effective fc = [{0}, {1}], cc=[{2}, {3}].'.format(fx, fy, cx, cy))
camera_matrix = np.eye(3)
camera_matrix[0, 0] = fx
camera_matrix[1, 1] = fy
camera_matrix[0, 2] = cx
camera_matrix[1, 2] = cy
distortion_params = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

os.system('mv %s %s' % (os.path.join(dataset_path, 'color'), os.path.join(dataset_path, 'color_raw')))

color_files = sorted(os.listdir(color_path))

if not os.path.exists(os.path.join(dataset_path, 'color')):
    os.makedirs(os.path.join(dataset_path, 'color'))
img_idx = 0

for i in range(0, len(color_files), 10):
    if "color_" in color_files[i]:
        color_idx = int(color_files[i][6:13])
        print(i, img_idx)
        I = cv2.imread(os.path.join(dataset_path, os.path.join('color_raw', color_files[i])))
        new_size = (I.shape[1] // downsample_factor, I.shape[0] // downsample_factor)
        I = cv2.resize(I, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        # undistort operation
        c_im = cv2.undistort(I, camera_matrix, distortion_params)
        C = cv2.resize(c_im[:, 120:840, :], (640, 480), cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dataset_path, 'color/color_%06d.png' % img_idx), C)
        img_idx += 1


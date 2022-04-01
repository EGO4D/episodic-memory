import numpy as np
import copy
import cv2
import os
import cv2
import numpy as np
import torch
import fnmatch
from torch.utils.data.dataset import Dataset
from SuperGlueMatching.models.utils import read_image

import open3d as o3d

def Vec2Skew(v):
    skew = np.asarray([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return skew


def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    q = np.empty([4,])

    tr = np.trace(R)
    if tr < 0:
        i = R.diagonal().argmax()
        j = (i + 1) % 3
        k = (j + 1) % 3

        q[i] = np.sqrt(1 - tr + 2 * R[i, i]) / 2
        q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
        q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
        q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    else:
        q[3] = np.sqrt(1 + tr) / 2
        q[0] = (R[2, 1] - R[1, 2]) / (4 * q[3])
        q[1] = (R[0, 2] - R[2, 0]) / (4 * q[3])
        q[2] = (R[1, 0] - R[0, 1]) / (4 * q[3])

    q /= np.linalg.norm(q)
    # Rearrange (x, y, z, w) to (w, x, y, z)
    q = q[[3, 0, 1, 2]]

    return q


def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R = np.empty([3, 3])
    R[0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[0, 1] = 2 * (x*y - z*w)
    R[0, 2] = 2 * (x*z + y*w)

    R[1, 0] = 2 * (x*y + z*w)
    R[1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[1, 2] = 2 * (y*z - x*w)

    R[2, 0] = 2 * (x*z - y*w)
    R[2, 1] = 2 * (y*z + x*w)
    R[2, 2] = 1 - 2 * x**2 - 2 * y**2

    return R

def WritePosesToPly(P, output_path):
    # Save the camera coordinate frames as meshes for visualization
    m_cam = None
    for j in range(P.shape[0]):
        # R_d = P[j, :, :3]
        # C_d = -R_d.T @ P[j, :, 3]
        T = np.eye(4)
        T[:3, :3] = P[j, :, :3].transpose()
        T[:3, 3] = -P[j, :, :3].transpose() @ P[j, :, 3]
        m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        m.transform(T)
        if m_cam is None:
            m_cam = m
        else:
            m_cam += m
    o3d.io.write_triangle_mesh(output_path, m_cam)


def skewsymm(x):
    S = np.zeros((x.shape[0], 3, 3))
    S[:, 0, 1] = -x[:, 2]
    S[:, 1, 0] = x[:, 2]
    S[:, 0, 2] = x[:, 1]
    S[:, 2, 0] = -x[:, 1]
    S[:, 1, 2] = -x[:, 0]
    S[:, 2, 1] = x[:, 0]

    return S


def VisualizeTriangulationMultiPoses(X, uv1, P, Im, K):
    out_img = []
    for i in range(P.shape[0]):
        kp_img = np.zeros_like(Im[i])
        # kpt = [cv2.KeyPoint(uv1[i, 0], uv1[i, 1], 20)]
        # cv2.drawKeypoints(Im[i], kpt, kp_img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img = cv2.drawMarker(Im[i], (int(uv1[i, 0]), int(uv1[i, 1])),
                       color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

        reproj = K @ P[i] @ np.hstack((X, np.array([1.0])))
        reproj = reproj / reproj[2]
        # reproj = [cv2.KeyPoint(reproj[0], reproj[1], 10)]
        # cv2.drawKeypoints(kp_img, reproj, kp_img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img = cv2.drawMarker(kp_img, (int(reproj[0]), int(reproj[1])),
                       color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
        out_img.append(kp_img)

    out_img = np.concatenate(out_img, axis=1)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 2, out_img.shape[0] // 2))
    cv2.imshow('kpts', small_out_image)
    cv2.waitKey(0)


def VisualizeMatches(im1_file, kp1, im2_file, kp2):
    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    cv_kp1 = [cv2.KeyPoint(x=kp1[i][0], y=kp1[i][1], _size=5) for i in range(len(kp1))]
    cv_kp2 = [cv2.KeyPoint(x=kp2[i][0], y=kp2[i][1], _size=5) for i in range(len(kp1))]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    out_img = np.zeros_like(im1, dtype=np.uint8)
    out_img = cv2.drawMatches(im1, cv_kp1, im2, cv_kp2, matches, None)
    cv2.imshow('matches', out_img)
    cv2.waitKey(0)


def VisualizeTrack(track, Im_input, K):
    Im = copy.deepcopy(Im_input)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    out_img = []
    trackx = track[:, 0] * fx + cx
    tracky = track[:, 1] * fy + cy

    for i in range(Im.shape[0]):
        if track[i, 0] == -1 and track[i, 1] == -1:
            out_img.append(Im[i])
        else:
            kp_img = cv2.drawMarker(Im[i], (int(trackx[i]), int(tracky[i])),
                                    color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

            # kp_img = cv2.drawMarker(kp_img, (int(reproj[0]), int(reproj[1])),
            #                         color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
            out_img.append(kp_img)

    out_img = np.concatenate(out_img, axis=1)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 4, out_img.shape[0] // 4))
    cv2.imshow('kpts', small_out_image)
    cv2.waitKey(0)


def VisualizeReprojectionError(P, X, track, K, Im):
    uv = track @ K.T
    kp_img = copy.deepcopy(Im)
    for i in range(uv.shape[0]):
        kp_img = cv2.drawMarker(kp_img, (int(uv[i, 0]), int(uv[i, 1])),
                       color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

    reproj = K @ P @ np.concatenate((X.T, np.ones((1, uv.shape[0]))), axis=0)
    reproj = reproj[:, reproj[2] > 0.3]
    reproj = reproj / reproj[2]
    for i in range(reproj.shape[1]):
        kp_img = cv2.drawMarker(kp_img, (int(reproj[0, i]), int(reproj[1, i])),
                       color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
    cv2.imshow('kpts', kp_img)
    cv2.waitKey(0)


def VisualizeBadPoseImage(Im):
    out_img = []
    num_images, h, w, _ = Im.shape
    num_image_per_row = 8
    rows = num_images // num_image_per_row
    left_over = num_images - num_image_per_row*rows
    for i in range(num_images//num_image_per_row):
        image_row = np.transpose(Im[num_image_per_row*i:num_image_per_row*(i+1)], (1, 0, 2, 3))
        image_row = image_row.reshape((h, num_image_per_row*w, 3))
        out_img.append(image_row)

    out_img = np.concatenate(out_img, axis=0)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 8, out_img.shape[0] // 8))
    cv2.imshow('bad pose images', small_out_image)
    cv2.waitKey(0)

## Below is for the visual database API

def convert_2d_to_3d(u, v, z, K):
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    return x, y, z


class MatterportDataset(Dataset):
    def __init__(self, dataset_folder='walterlib', resize=[640, 480]):
        super(MatterportDataset, self).__init__()
        self.dataset_folder = dataset_folder
        self.data_path = os.path.join(self.dataset_folder, 'color')
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
    def __init__(self, dataset_folder='walter_basement_03', resize=[640, 480],
                 start_idx=0, end_idx=10000, skip_every_n_image=1):
        super(AzureKinect, self).__init__()
        self.dataset_folder = dataset_folder
        self.data_path = os.path.join(self.dataset_folder, 'color')
        self.data_info = sorted(fnmatch.filter(os.listdir(self.data_path), '*.jpg'))
        self.data_info = self.data_info[start_idx:end_idx:skip_every_n_image]
        self.start_idx = start_idx
        self.data_len = len(self.data_info)
        self.resize = resize

    def __getitem__(self, index):
        # color_info = os.path.join(self.data_path, 'color_%07d.jpg' % (self.start_idx + index))
        color_info = os.path.join(self.data_path, self.data_info[index])
        # print(color_info)
        _, gray_tensor, _ = read_image(color_info, 'cpu', resize=self.resize, rotation=0, resize_float=False)
        gray_tensor = gray_tensor.reshape(1, 480, 640)

        output = {'image': gray_tensor, 'image_index': int(color_info[-11:-4])}

        return output

    def __len__(self):
        return self.data_len

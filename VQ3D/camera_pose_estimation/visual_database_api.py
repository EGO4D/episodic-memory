import torch
import cv2
import fnmatch

from utils import *
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

from SuperGlueMatching.models.utils import read_image
from SuperGlueMatching.models.superpoint import SuperPoint
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm


parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--matterport_descriptors_folder', type=str, default='',
    help='Matterport data set')
parser.add_argument(
    '--matterport_output_folder', type=str, default='',
    help='Matterport data set')
parser.add_argument(
    '--ego_dataset_folder', type=str, default='',
    help='Ego data set')

opt = parser.parse_args()

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }
}

IMAGE_DESC_FOLDER = opt.matterport_descriptors_folder
OUTPUT_FOLDER = opt.matterport_output_folder

assert (os.path.exists(IMAGE_DESC_FOLDER))

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

K_mp = np.array([[700., 0., 960. - 0.5],
                 [0., 700., 540. - 0.5],
                 [0., 0., 1.]])

superpoint = SuperPoint(config.get('superpoint', {})).cuda()
superpoint = superpoint.eval()

############# STEP 0: CONSTRUCT DESCRIPTOR CENTERS
print('Compute descriptor centroids with kmeans++ ... ')

if os.path.exists(os.path.join(OUTPUT_FOLDER, 'data_descriptors_centers.npy')):
    descriptors_cluster_centers = np.load(os.path.join(OUTPUT_FOLDER,
                                                       'data_descriptors_centers.npy'),'r')
else:
    total_removal = 0
    all_descriptors = None
    
    with torch.no_grad():
        # all the image descriptors we got to build the visual database
        matterport_filelist = sorted(fnmatch.filter(os.listdir(IMAGE_DESC_FOLDER), 'image_*_descriptors.npz'))
        for i, filename in enumerate(tqdm(matterport_filelist)):
            image_descriptors = np.load(os.path.join(IMAGE_DESC_FOLDER, filename),'r')['descriptors']
            _, N = image_descriptors.shape
            if N < 300:
                total_removal += 1
                continue
            if all_descriptors is None:
                all_descriptors = np.transpose(image_descriptors, (1, 0))
            else:
                all_descriptors = np.concatenate((all_descriptors,
                                              np.transpose(image_descriptors, (1, 0))), axis=0)

    kmeans = KMeans(n_clusters=64, random_state=0, init='k-means++', max_iter=5000, verbose=0).fit(all_descriptors)  # data
    np.save(os.path.join(OUTPUT_FOLDER, 'data_descriptors_centers.npy'), kmeans.cluster_centers_)
    descriptors_cluster_centers = kmeans.cluster_centers_


############# STEP 1: BUILD IMAGE DESCRIPTOR TREE
descriptors_cluster_centers = torch.tensor(descriptors_cluster_centers, device='cuda')  # KxD
K, D = descriptors_cluster_centers.shape

image_indices = []
vlad_image_descriptors = []

print('Compute image descriptor for each Matterport image ... ')
if os.path.exists(os.path.join(OUTPUT_FOLDER, 'all_image_descriptors.npy')):
    vlad_image_descriptors = np.load(os.path.join(OUTPUT_FOLDER,
                                                  'all_image_descriptors.npy'),'r')
    image_indices = np.load(os.path.join(OUTPUT_FOLDER, 'all_image_descriptors_indices.npy'),'r')
else:
    with torch.no_grad():
        matterport_filelist = sorted(fnmatch.filter(os.listdir(IMAGE_DESC_FOLDER), 'image_*_descriptors.npz'))

        for filename in tqdm(matterport_filelist):
            image_descriptors = np.load(os.path.join(IMAGE_DESC_FOLDER,
                                                     filename),'r')['descriptors']
            image_descriptors = torch.tensor(image_descriptors, device='cuda')
            _, N = image_descriptors.shape
            if N < 250:
                continue

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

            image_indices.append(int(filename[6:12]))
            vlad_image_descriptors.append(vlad_image_descriptor.detach().cpu().numpy())

        vlad_image_descriptors = np.concatenate(vlad_image_descriptors, axis=0)
        image_indices = np.asarray(image_indices)
        
        np.save(os.path.join(OUTPUT_FOLDER, 'all_image_descriptors.npy'), vlad_image_descriptors)
        np.save(os.path.join(OUTPUT_FOLDER, 'all_image_descriptors_indices.npy'), image_indices)


# ######### STEP 2: QUERY IMAGE FROM DATABASE
kdt = KDTree(vlad_image_descriptors, leaf_size=30, metric='euclidean')

EGO_DATASET_FOLDER = opt.ego_dataset_folder

if not os.path.exists(os.path.join(EGO_DATASET_FOLDER, 'vlad_best_match')):
    os.makedirs(os.path.join(EGO_DATASET_FOLDER, 'vlad_best_match'))

ego_dataset = AzureKinect(dataset_folder=EGO_DATASET_FOLDER, 
                          skip_every_n_image=1,
                          start_idx=0, 
                          end_idx=-1)
data_loader = DataLoader(dataset=ego_dataset,
                         num_workers=4, 
                         batch_size=16,
                         shuffle=False,
                         pin_memory=True)

print('Query image with database images')
match_pair = {'matterport': [], 'ego': []}
with torch.no_grad():
    for idx, images in enumerate(tqdm(data_loader)):
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

            for qids in range(len(ret_query)):
                match_pair['matterport'].append(
                    os.path.join(IMAGE_DESC_FOLDER, 'image_%06d_descriptors.npz' % (image_indices[ret_query[qids]])))
                match_pair['ego'].append(
                    os.path.join(EGO_DATASET_FOLDER, 'color', 'color_%07d.jpg' % images['image_index'][j]))

with open(os.path.join(EGO_DATASET_FOLDER, 'vlad_best_match', 'queries.pkl'), 'wb') as f:
    pickle.dump(match_pair, f)

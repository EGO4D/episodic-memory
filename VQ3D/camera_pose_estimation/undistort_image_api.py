from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
import argparse
import fnmatch
from tqdm import tqdm
from utils import *

to_tensor = transforms.ToTensor()

parser = argparse.ArgumentParser(
        description='Undistort image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--ego_dataset_folder', type=str, default='',
    help='Azure dataset folder')
parser.add_argument(
    '--crop_x', type=int, default=0,
    help='Crop image x dim -crop_x : crop_x')
parser.add_argument(
    '--crop_y', type=int, default=0,
    help='Crop image y dim -crop_y : crop_y')


opt = parser.parse_args()

ROOT_FOLDER = os.path.join(opt.ego_dataset_folder, 'color_distorted')
OUT_FOLDER = os.path.join(opt.ego_dataset_folder, 'color')
camera_intrinsics = np.loadtxt(os.path.join(opt.ego_dataset_folder, 'fisheye_intrinsics.txt'))
W = int(camera_intrinsics[0])
H = int(camera_intrinsics[1])
cx = camera_intrinsics[2]
cy = camera_intrinsics[3]
f = camera_intrinsics[4]
k1 = camera_intrinsics[5]
k2 = camera_intrinsics[6]

if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

for img_idx in tqdm(range(len(fnmatch.filter(os.listdir(ROOT_FOLDER), '*.jpg')))):
    image = Image.open(os.path.join(ROOT_FOLDER, 'color_%07d.jpg' % img_idx))
    image = to_tensor(image)
    image = image.reshape(1, 3, H, W)

    [YY, XX] = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    ## Radial Fisheye
    UU = 2.5 / f * (XX - cx)
    VV = 2.5 / f * (YY - cy)
    r = torch.sqrt(UU * UU + VV * VV)
    valid_r_mask = (r > 1e-4)

    du = torch.zeros_like(r)
    dv = torch.zeros_like(r)
    tt = torch.atan(r[valid_r_mask])
    tt2 = tt * tt
    tt4 = tt2 * tt2
    ttd = tt * (1.0 + k1 * tt2 + k2 * tt4)
    du[valid_r_mask] = UU[valid_r_mask] * (ttd / r[valid_r_mask] - 1)
    dv[valid_r_mask] = VV[valid_r_mask] * (ttd / r[valid_r_mask] - 1)

    UU_dist = UU + du
    VV_dist = VV + dv

    XX_dist = f * UU_dist + cx
    YY_dist = f * VV_dist + cy


    ### Sampling descriptor
    grid_sampler = torch.zeros((1, H, W, 2))
    grid_sampler[0, :, :, 0] = 2.0 / W * (XX_dist - cx)
    grid_sampler[0, :, :, 1] = 2.0 / H * (YY_dist - cy)

    undistorted_image = torch.nn.functional.grid_sample(image, 
                                                        grid_sampler, 
                                                        mode='bilinear', 
                                                        padding_mode='zeros',
                                                        align_corners=True)
    undistorted_image = 255. * torch.permute(undistorted_image[0, :, opt.crop_y:-opt.crop_y, opt.crop_x:-opt.crop_x], (1, 2, 0)).cpu().numpy()

    undistorted_image = Image.fromarray(undistorted_image.astype(np.uint8))
    undistorted_image.save(os.path.join(OUT_FOLDER, 'color_%07d.jpg' % img_idx))

K = np.array([[f/2.5, 0., (W-2.*opt.crop_x)/2. - 0.5],
              [0., f/2.5, (H-2.*opt.crop_y)/2. - 0.5],
              [0., 0., 1.]])

np.savetxt('%s/intrinsics.txt' % opt.ego_dataset_folder, K)

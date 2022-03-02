import os
import sys
import cv2
import json
import fnmatch
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox


class VisualQuery3DGroundTruth():
    def __init__(self):
        pass

    def load_pose(self, dirname: str):
        pose_dir = os.path.join(dirname, 'superglue_track', 'poses')

        if not os.path.isfile(os.path.join(pose_dir,
                                           'cameras_pnp_triangulation.npy')):
            return None

        original_image_ids = np.arange(0,len(fnmatch.filter(os.listdir(os.path.join(dirname,
                                                                                     'color')),
                                                             '*.jpg')))

        valid_pose = np.load(os.path.join(pose_dir, 'good_pose_reprojection.npy'))
        C_T_G = np.load(os.path.join(pose_dir, 'cameras_pnp_triangulation.npy'))

        Ci_T_G = np.zeros((len(original_image_ids), 4, 4))
        k = 0
        for i in range(len(original_image_ids)):
            if valid_pose[i]:
                Ci_T_G[k] = np.concatenate((C_T_G[i], np.array([[0., 0., 0., 1.]])), axis=0)
                k += 1
            else:
                Ci_T_G[k] = np.eye(4)
                Ci_T_G[k][2, 3] = 100
                k += 1

        return Ci_T_G, valid_pose

    def load_3d_annotation(self, data: Dict):
        # use Ego4D-3D-Annotation API
        box = BoundingBox()
        box.load(data)
        return box.center


    def create_traj_azure(self, output_traj, K, Ci_T_G=None):

        d = json.load(open('../camera_pose_estimation/Visualization/camera_trajectory.json', 'r'))
        dp0 = d['parameters'][0]
        dp0['intrinsic']['width'] = int((K[6]+0.5)*2)
        dp0['intrinsic']['height'] = int((K[7]+0.5)*2)
        dp0['intrinsic']['intrinsic_matrix'] = K.tolist()
        dp0['extrinsic'] = []
        x = []


        if Ci_T_G is not None:
            for i in range(Ci_T_G.shape[0]):
                temp = dp0.copy()

                # E = np.linalg.inv(G_T_Ci[i])
                E = Ci_T_G[i]

                E_v = np.concatenate([E[:, i] for i in range(4)], axis=0)
                temp['extrinsic'] = E_v.tolist()
                x.append(temp)

        d['parameters'] = x
        with open(output_traj, 'w') as f:
            json.dump(d, f)



    def read_pfm(self, path):
        import sys
        import re
        import numpy as np
        import cv2
        import torch

        from PIL import Image
        """Read pfm file.
        Args:
            path (str): path to file
        Returns:
            tuple: (data, scale)
        """
        with open(path, "rb") as file:

            color = None
            width = None
            height = None
            scale = None
            endian = None

            header = file.readline().rstrip()
            if header.decode("ascii") == "PF":
                color = True
            elif header.decode("ascii") == "Pf":
                color = False
            else:
                raise Exception("Not a PFM file: " + path)

            dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
            if dim_match:
                width, height = list(map(int, dim_match.groups()))
            else:
                raise Exception("Malformed PFM header.")

            scale = float(file.readline().decode("ascii").rstrip())
            if scale < 0:
                # little-endian
                endian = "<"
                scale = -scale
            else:
                # big-endian
                endian = ">"

            data = np.fromfile(file, endian + "f")
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)

            return data, scale



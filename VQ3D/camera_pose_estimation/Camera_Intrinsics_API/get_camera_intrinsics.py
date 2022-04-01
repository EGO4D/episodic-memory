import os
import cv2
import shutil
import numpy as np
import subprocess as sp
from imageio import imread

from typing import Any, Dict, List, Optional, Tuple


class CameraIntrinsicsHelper():
    def __init__(self):
        self.blurry_thresh = 100.0
        self.sfm_workspace_dir = 'data/debug_sfm/'
        self.sfm_images_dir = 'data/debug_sfm/images'

    def is_blurry(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var > self.blurry_thresh:
            # not blurry
            return False, var
        else:
            return True, var

    def select_good_frames_indices(self, images_files: List, images_dir: str) -> Tuple:
        blurry_indicator = []
        all_laplacian_variances = []
        cpt_consecutive_non_blurry = 0
        for image_file in images_files:
            image = imread(os.path.join(images_dir, image_file))
            blurry_ind_img, tmp_var = self.is_blurry(image)
            blurry_indicator.append(blurry_ind_img)
            all_laplacian_variances.append(tmp_var)
            if not blurry_ind_img:
                cpt_consecutive_non_blurry+=1
            else:
                cpt_consecutive_non_blurry=0
            if cpt_consecutive_non_blurry>10:
                break
        
        blurry_indicator = np.array(blurry_indicator)
        blurry_indicator = blurry_indicator.astype(np.int)

        diff = np.diff(blurry_indicator)
        nonzero_indices = np.nonzero(diff!=0)[0]

        if len(nonzero_indices) > 0:
            splits = np.split(blurry_indicator, nonzero_indices+1)
            idx = 0
            max_idx_start = -1
            max_idx_end = -1
            max_length = -1
            for s in splits:
                if s.size==0: continue
                if s[0] == 0:
                    assert np.all(s==0)
                    if len(s) > max_length:
                        max_length = len(s)
                        max_idx_start = idx
                        max_idx_end = idx + len(s) - 1
                idx += len(s)

            if max_idx_start > -1:
                start_idx = max_idx_start
                end_idx = max_idx_end
                # check the number of frame selected.
                # if not enough frames are selected.
                # get the set of 10 frames with the 
                # highest cumulative laplacian variance
                if (end_idx - start_idx + 1) < 5:
                    print('Not enough frames selected -> Getting the set of 10 frames with max cumsum Laplacian Variance')
                    cumsum_10frames = [np.sum(all_laplacian_variances[i:i+10]) for i\
                                      in range(0,len(all_laplacian_variances)-10)]
                    idx = np.argmax(cumsum_10frames)
                    start_idx = idx
                    end_idx = idx+9
                
                return (start_idx, end_idx)
            else:
                return None
        else:
            if blurry_indicator[0] == 0:
                #all frames are not blurry
                max_idx = 0
                max_length = len(blurry_indicator)
                start_idx = int(len(blurry_indicator) / 2.0)
                end_idx = int(len(blurry_indicator) / 2.0) + 9
                return (start_idx, end_idx)
            else:
                # all frames are blurry
                # then select the most non-blurry ones
                print('ALL frames are blurry -> Getting the set of 10 frames with max cumsum Laplacian Variance')
                all_laplacian_variances = np.array(all_laplacian_variances)
                cumsum_10frames = [np.sum(all_laplacian_variances[i:i+10]) for i\
                                  in range(0,len(all_laplacian_variances)-10)]
                idx = np.argmax(cumsum_10frames)
                start_idx = idx
                end_idx = idx+9
                return (start_idx, end_idx)


    def select_good_frames(self, images_dir:str) -> None:
        r"""
        Selects good frames for intrinsics parameter estimation using COLMAP.
        A good frame is non-blury - it has a variance of Laplacian greater than 100. 
        We aim at selecting 20 consecutive of such good frames for a better
        COLMAP performance.
        """
        images_files = os.listdir(images_dir)
        images_files = sorted(images_files)
        indices = self.select_good_frames_indices(images_files, images_dir)
        if indices is not None:
            cpt = 0
            for i in range(indices[0], indices[1]+1, 1):
                shutil.copyfile(os.path.join(images_dir, images_files[i]),
                                os.path.join(self.sfm_images_dir, images_files[i]))
                cpt+=1
                if cpt > 10: break
        else:
            print('NO GOOD FRAMES FOUND')


    def run_colmap(self) -> List:
        # run sfm
        o = sp.check_output(['colmap',
                             "automatic_reconstructor",
                             "--workspace_path",
                             self.sfm_workspace_dir,
                             "--image_path",
                             self.sfm_images_dir,
                             "--camera_model",
                             "RADIAL_FISHEYE",
                             "--single_camera",
                             "1",
                            ])

        # check if successfull
        if len(os.listdir(os.path.join(self.sfm_workspace_dir,'sparse'))) == 0:
            return None

        o = sp.check_output(['colmap',
                             "model_converter",
                             "--input_path",
                             os.path.join(self.sfm_workspace_dir, 'sparse', '0/'),
                             "--output_path",
                             os.path.join(self.sfm_workspace_dir, 'sparse', '0/'),
                             "--output_type",
                             "TXT",
                            ])
        
        return self.parse_colmap_intrinsics(os.path.join(self.sfm_workspace_dir, 'sparse/0/cameras.txt'))


    def parse_colmap_intrinsics(self, camera_txt_filename: str)-> Dict:
        # example output
        #1 RADIAL_FISHEYE 1440 1080 660.294 720 540 0.0352702 0.0046637\n
        with open(camera_txt_filename, 'r') as f:
            for _ in range(4):
                l = f.readline()
        e = l.split(' ')
        outputs = {
            'num_cameras': e[0],
            'type':e[1],
            'width':e[2],
            'height':e[3],
            'f':e[4],
            'cx':e[5],
            'cy':e[6],
            'k1':e[7],
            'k2':e[8][:-1],
        }
        return outputs


    def get_camera_intrinsics(self, images_dir: str) -> Tuple:
        self.select_good_frames(images_dir)
        return self.run_colmap()



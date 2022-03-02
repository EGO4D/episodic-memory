import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import json
import h5py
import argparse
from PIL import Image
import numpy as np
import open3d as o3d
import glob
import cv2
import fnmatch
from imageio import imwrite


W = 960
H = 540

def create_traj_azure(output_traj, K, Ci_T_G=None):

    d = json.load(open('Visualization/camera_trajectory.json', 'r'))
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


def custom_draw_geometry_with_camera_trajectory(pcd, output_path='',
                                                output_path_depth='',
                                                input_image_folder='',
                                                comparison_color='',
                                                trajectory='',
                                                G_T_Ci=None,
                                                K=None,
                                                original_img_indices=None,
                                                valid_pose=None):

    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = \
        o3d.io.read_pinhole_camera_trajectory(trajectory)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    def setup(vis):
        ctr = vis.get_view_control()
        #ctr.set_zoom(0.450)
        ctr.rotate(0.0, -4e2)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        vis.update_geometry(None)

        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory

        if glb.index >= 0 and glb.index < len(original_img_indices):
            if valid_pose[glb.index]:
                print("Capture image {:05d}".format(glb.index))
                image = vis.capture_screen_float_buffer(False)  # True for
                image = np.asarray(image)
                depth = vis.capture_depth_float_buffer(False)
                depth = np.asarray(depth)
                if output_path_depth != '':
                    filename = os.path.join(output_path_depth,  'render_%07d.h5' % glb.index)
                    with h5py.File(filename, 'w') as f:
                        f.create_dataset('depth', data=depth, dtype=np.float32)

                    if np.max(depth) >0:
                        depth = depth / np.max(depth) * 255.0
                    depth = depth.astype(np.uint8)
                    imwrite(os.path.join(output_path_depth, 'render_%07d.png'%glb.index), depth)

                if output_path != '':
                    color_info = os.path.join(input_image_folder, 'color/color_%07d.jpg' % original_img_indices[glb.index])
                    color_img = Image.open(color_info)
                    # color_img = color_img.resize((960, 540), resample=Image.BILINEAR)

                    img = (255. * image)
                    added_image = cv2.addWeighted(np.asarray(color_img), 0.8, img.astype(np.uint8), 0.5, 0)
                    full_viz_image = np.concatenate([np.asarray(color_img), img.astype(np.uint8), added_image], axis=1)
                    cv2.imwrite(os.path.join(output_path, 'render_%07d.png' % glb.index),
                                cv2.cvtColor(full_viz_image, cv2.COLOR_RGB2BGR))

        glb.index += 1

        if glb.index < len(original_img_indices): # visualize only well estimated views
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            exit(1)

        return False


    W = int((K[0,2]+0.5)*2)
    H = int((K[1,2]+0.5)*2)

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=W, height=H, visible=False) # visible = False
    vis.add_geometry(pcd)
    setup(vis)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Incremental SFM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--matterport_dataset_folder', type=str, default='',
        help='Matterport dataset folder')
    parser.add_argument(
        '--ego_dataset_folder', type=str, default='',
        help='Ego dataset folder')

    opt = parser.parse_args()



    ROOT_FOLDER = opt.ego_dataset_folder
    INPUT_DIR = os.path.join(ROOT_FOLDER, 'superglue_track', 'poses')

    original_image_ids = np.arange(0, len(fnmatch.filter(os.listdir(os.path.join(ROOT_FOLDER, 'color')), '*.jpg')))

    valid_pose = np.load(os.path.join(INPUT_DIR, 'good_pose_reprojection.npy'))
    #valid_pose = np.load(os.path.join(INPUT_DIR, 'good_poses_ids_pnp_triangulation.npy'))
    C_T_G = np.load(os.path.join(INPUT_DIR, 'cameras_pnp_triangulation.npy'))


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

    print("Create trajectory ...")
    K = np.loadtxt('%s/intrinsics.txt' % ROOT_FOLDER)
    K_v = np.concatenate([K[:, i] for i in range(3)], axis=0)
    W = int((K[0,2]+0.5)*2)
    H = int((K[1,2]+0.5)*2)

    create_traj_azure(output_traj='Visualization/egovideo_camera_traj.json',
                      K=K_v,
                      Ci_T_G=Ci_T_G)

    print("Loading point cloud ...")
    SAVING_SUBFOLDER = 'pose_visualization_superglue'
    SAVING_SUBFOLDER_DEPTH = 'pose_visualization_depth_superglue'
    if not os.path.exists(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER)):
        os.makedirs(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER))
    if not os.path.exists(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER_DEPTH)):
        os.makedirs(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER_DEPTH))

    mesh_file = fnmatch.filter(os.listdir(os.path.join(opt.matterport_dataset_folder, 'matterpak')), '*.obj')[0]
    mesh = o3d.io.read_triangle_mesh(os.path.join(opt.matterport_dataset_folder,
                                                  'matterpak', mesh_file))
    custom_draw_geometry_with_camera_trajectory(mesh,
                                                output_path=os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER),
                                                output_path_depth=os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER_DEPTH),
                                                input_image_folder=ROOT_FOLDER,
                                                trajectory='./Visualization/egovideo_camera_traj.json',
                                                G_T_Ci=Ci_T_G,
                                                K=K,
                                                original_img_indices=original_image_ids,
                                                valid_pose=valid_pose)

# STEP 0 [Optional]: Extract keypoints and visualize it
# python ./SuperGlueMatching/extract_keypoints.py

# STEP 1: Extract keypoints and visualize it
# This script has 2 functionality:
# (i) It saves all Matterport vlad image descriptors and theirs indices
# vlad_image_descriptors = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors.npy'))
# image_indices = np.load(os.path.join(ROOT_DIR, DATASET_NAME, 'all_image_descriptors_indices.npy'))
# (ii) Using the previous created descriptors, create a tree and query AzureKinect images, top matches are saved
# np.savetxt(os.path.join(AZURE_ROOT_DIR, AZURE_DATASET_NAME, 'vlad_best_match', 'queries_%06d.txt' % images['image_index'][j]),
#                       np.array(image_indices[ret_query]), fmt='%d')
python ./SuperGlueMatching/extract_and_cluster_descriptors.py

# STEP 2: Run SuperGlue matching based on the top queries
# TODO: Change output dir to ~/Data/KiNectAzure/....
# TODO: Merge this with step 4 - match and get poses
# TODO: Parallelization - dataloader and match in batch
python ./SuperGlueMatching/match_pairs.py \
--input_pairs '/media/tiendo/DATA/KinectAzure/walterb18/vlad_best_match/queries.pkl' \
--starting_index 0 --ending_index -1 \
--input_dir '/media/tiendo/DATA/' \
--output_dir '/media/tiendo/DATA/KinectAzure/walterb18/superglue_match_results/' \
--viz

# STEP 3: Run pnp to determine poses
# This scripts reads matches from previous step and save poses to ~/Data/KiNectAzure/.../pose_pred_pnp/
# Identity poses are saved if RANSAC returns unreliable solution
# TODO: Parallelization - pool?
python ./Localization/test_multi_pnps.py

# STEP 4: Visualize overlaid-point-cloud-projected image on actual Azure image
# Output is saved in ~/Data/KiNectAzure/.../overlaid/
python ./Localization/visualize_pcd_pose_rendered.py

# STEP 5: Final visualization, pose on map, overlaid image, and matching/query
# Right now, output is saved in ~/Data/KiNectAzure/.../full_viz/
# There is an issue with *10
matlab ./Visualization/visualize_localization_on_3dmap.m

# STEP 6: Create videos
./Visualization/make_video.sh
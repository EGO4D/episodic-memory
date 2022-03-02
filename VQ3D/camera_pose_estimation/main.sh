#!/bin/bash

POSITIONAL_ARGS=()

UNDISTORT=false
DATABASE=false
PNP=false
TRACK=false
SFM=false
VIZ=false

while [[ $# -gt 0 ]]; do
    case $1 in
      --undistort)
        UNDISTORT=true
        shift
        ;;
      --database)
        DATABASE=true
        shift
        ;;
      --pnp)
        PNP=true
        shift
        ;;
      --track)
        TRACK=true
        shift
        ;;
    --sfm)
        SFM=true
        shift
        ;;
    --viz)
        VIZ=true
        shift
        ;;
      *)
        POSITIONAL_ARGS+=($1)
        shift
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ "$UNDISTORT" = true ] ; then
    echo "Undistort image"
    python undistort_image_api.py \
    --ego_dataset_folder $2 --crop_x 300 --crop_y 300
fi

if [ "$DATABASE" = true ] ; then
    echo "Extract image descriptor and query"
    python visual_database_api.py \
    --matterport_descriptors_folder $1/descriptors/ \
    --matterport_output_folder $1/descriptors/ \
    --ego_dataset_folder $2
fi

if [ "$PNP" = true ] ; then
    echo "Match query and database images"
    python SuperGlueMatching/match_pairs_api.py \
    --input_pairs $2/vlad_best_match/queries.pkl \
    --starting_index 0 --ending_index 0 \
    --output_dir $2/superglue_match_results/
    
    echo "Obtain poses with PnP"
    python pnp_api.py \
    --ego_dataset_folder $2 \
    --matterport_descriptors_folder $1/descriptors/ \
    --output_dir $2/poses_reloc/
fi

if [ "$TRACK" = true ] ; then
    echo "Extract temporal constraints by matching pairwise images"
    python superglue_tracker.py \
    --ego_dataset_folder $2 \
    --extract_descriptor
fi

if [ "$SFM" = true ] ; then
    echo "Incremental unordered sfm"
    python sfm_api_wsuperglue.py \
    --ego_dataset_folder $2
fi

if [ "$VIZ" = true ] ; then
    echo "Image overlaid visualization"
    python Visualization/visualize_render_images.py \
    --matterport_dataset_folder $1 \
    --ego_dataset_folder $2
fi

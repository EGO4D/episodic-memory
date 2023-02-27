#!/bin/bash

source ~/enable_em_vq2d.sh

SPLIT=$1
MODEL_ROOT=$2
CACHE_ROOT=$3
VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"
CLIPS_ROOT="$VQ2D_ROOT/data/clips"

cd $VQ2D_ROOT

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"

python -W ignore extract_vq_detection_scores.py \
  data.data_root="$CLIPS_ROOT" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.split="$SPLIT" \
  data.num_processes_per_gpu=2 \
  data.rcnn_batch_size=10 \
  model.config_path="$MODEL_ROOT/config.yaml" \
  model.checkpoint_path="$MODEL_ROOT/model.pth" \
  model.cache_root="$CACHE_ROOT"

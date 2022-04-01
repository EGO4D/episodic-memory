#!/bin/bash

module purge

EXPT_ROOT=$PWD

module load anaconda3/2020.11
module load cuda/10.2
module load cudnn/v7.6.5.32-cuda.10.2
module load gcc/7.3.0
module load cmake/3.15.3/gcc.7.3.0

source activate ego4d_vq2d

CLIPS_ROOT="<PATH TO CLIPS ROOT DIRECTORY>"
VQ2D_ROOT="<PATH TO VQ2D>"
VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"

cd $VQ2D_ROOT

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"
export PATH="<PATH to conda environment binaries>:$PATH"


python evaluate_vq2d.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=2 \
  model.config_path="$EXPT_ROOT/logs/config.yaml" \
  model.checkpoint_path="$EXPT_ROOT/logs/model_<CHECKPOINT_ID>.pth" \
  logging.save_dir="$EXPT_ROOT/visual_queries_logs" \
  logging.stats_save_path="$EXPT_ROOT/visual_queries_logs/vq_stats.json.gz"
#!/bin/bash

source ~/enable_em_vq2d.sh

MODEL_ROOT=$1
CACHE_ROOT=$2
SPLIT=$3
N_PROCS_PER_GPU=$4
PEAK_SIMILARITY_THRESH=$5
LOST_THRESH=$6

VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"
CLIPS_ROOT="$VQ2D_ROOT/data/clips"
STATS_PATH=$MODEL_ROOT/${SPLIT}_predictions_pst_${PEAK_SIMILARITY_THRESH}_lost_thresh_${LOST_THRESH}.json

cd $VQ2D_ROOT

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"

NT=1
OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT VECLIB_MAXIMUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT python -W ignore perform_vq_inference.py \
    data.data_root=$CLIPS_ROOT \
    data.annot_root=$VQ2D_SPLITS_ROOT \
    data.split=$SPLIT \
    data.num_processes_per_gpu=$N_PROCS_PER_GPU \
    data.rcnn_batch_size=1 \
    model.config_path=$MODEL_ROOT/config.yaml \
    model.checkpoint_path=$MODEL_ROOT/model.pth \
    model.cache_root=$CACHE_ROOT \
    signals.peak_similarity_thresh=$PEAK_SIMILARITY_THRESH \
    logging.save_dir=$MODEL_ROOT \
    logging.stats_save_path=$STATS_PATH \
    tracker.kys_tracker.model_path=$VQ2D_ROOT/pretrained_models/kys.pth \
    tracker.kys_tracker.lost_thresh=$LOST_THRESH

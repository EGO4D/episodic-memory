#!/bin/bash

module purge

ROOT_DIR=$PWD

module load anaconda3/2020.11
module load cuda/10.2
module load cudnn/v7.6.5.32-cuda.10.2
module load gcc/7.3.0
module load cmake/3.15.3/gcc.7.3.0

source activate ego4d_vq2d

export PYTHONPATH=$PYTHONPATH:<PATH to VQ2D directory>
export PATH=<PATH to conda environment binaries>:$PATH

cd <PATH TO VQ2D>


python train_siam_rcnn.py \
    --num-gpus 8 \
    --config-file configs/siam_rcnn_8_gpus.yaml \
    --resume \
    OUTPUT_DIR $ROOT_DIR/logs \
    INPUT.VQ_IMAGES_ROOT <PATH TO IMAGES ROOT DIRECTORY> \
    INPUT.VQ_DATA_SPLITS_ROOT <PATH TO VQ2D>/data
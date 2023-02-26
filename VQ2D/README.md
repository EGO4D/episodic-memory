# Visual Queries 2D localization

## Installation

1. Clone the repository from [here](https://github.com/EGO4D/episodic-memory).
    ```
    git clone git@github.com:EGO4D/episodic-memory.git
    cd episodic-memory/VQ2D
    export VQ2D_ROOT=$PWD
    ```
2. Create conda environment.
    ```
    conda create -n ego4d_vq2d python=3.8
    ```

3. Install [pytorch](https://pytorch.org/) using conda. We rely on cuda-10.2 and cudnn-7.6.5.32 for our experiments.
    ```
    conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    ```

4. Install additional requirements using `pip`.
    ```
    pip install -r requirements.txt
    ```

5. Install [detectron2](https://github.com/facebookresearch/detectron2).
    ```
    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
    ```

6.  Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQ2D_ROOT/pretrained_models/kys.pth`.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:visionml/pytracking.git
    git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
    ```

    Note: For installing the [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) dependency for pytracking, follow these steps if the pip install fails.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
    cd Pytorch-Correlation-extension
    python setup.py install
    ```
7. Create a script `~/enable_em_vq2d.sh` to set necessary environment variables and activate the conda environment.
    ```
    #!/usr/bin/bash

    # Add anaconda path
    export PATH="$PATH:<PATH TO anaconda3>/bin"
    # Activate conda environment
    source activate ego4d_vq2d

    CUDA_DIR=<PATH TO cuda-10.2>
    CUDNN_DIR=<PATH TO cudnn-10.2-v8.0.3>

    # Add cuda, cudnn paths
    export CUDA_HOME=$CUDA_DIR
    export CUDNN_PATH=$CUDNN_DIR/cuda/lib64/libcudnn.so
    export CUDNN_INCLUDE_DIR=$CUDNN_DIR/cuda/include
    export CUDNN_LIBRARY=$CUDNN_DIR/cuda/lib64
    export CUDACXX=$CUDA_DIR/bin/nvcc

    export VQ2D_ROOT=<PATH TO episodic-memory repo>/VQ2D"
    ```

## Preparing data for training and inference

1. Download the videos as instructed [here](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to `$VQ2D_ROOT/data`.
    ```
    ego4d --output_directory="$VQ2D_ROOT/data" --datasets full_scale
    # Define ego4d videos directory
    export EGO4D_VIDEOS_DIR=$VQ2D_ROOT/data/v1/full_scale
    ```
2. Download the latest annotations to `$VQ2D_ROOT/data`. We use an updated version (v1.0.5) of the VQ2D annotations which includes fixes to a subset of data (check details [here](https://eval.ai/web/challenges/challenge-page/1843/overview)). These primarily affect the train and val splits (and not test split). In local experiments, we find that this leads to improved baseline performance on the val split:
    ```
    # Download the data using the Ego4D CLI.
    ego4d --output_directory="$VQ2D_ROOT/data" --datasets annotations -y --version v2

    # Move out vq annotations to $VQ2D_ROOT/data
    mv $VQ2D_ROOT/data/v2/annotations/vq_*.json $VQ2D_ROOT/data
    ```

3. Process the VQ dataset.
    ```
    python process_vq_dataset.py --annot-root data --save-root data
    ```

4. Extract clips for val and test data from videos. Validate the clips once they are extracted. If validation fails, please re-run the conversion script and it will correct for errors. You can optionally add a `--clip-uids <clip-uid-1> <clip-uid-2> ...` argument to specify the clips to regenerate. You can optionally reduce the video frame resolution by specifying `--downscale-height <height>`.
    ```
    # Extract clips (should take 12-24 hours on a machine with 80 CPU cores)
    python convert_videos_to_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --save-root data/clips \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed

    # Validate the extracted clips (should take 30 minutes)
    python tools/validate_extracted_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --clips-root data/clips
    ```

5. Extract images for train and validation data from videos (only needed for training detection models).
    ```
    # Should take <= 6 hours on a machine with 80 CPU cores
    python convert_videos_to_images.py \
        --annot-paths data/vq_train.json data/vq_val.json \
        --save-root data/images \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed
    ```

## Training detection models
Copy `scripts/train_2_gpus.sh` or `scripts/train_8_gpus.sh` to the required experiment directory and execute it.

```
EXPT_ROOT=<experiment path>
cp $VQ2D_ROOT/scripts/train_2_gpu.sh $EXPT_ROOT
cd $EXPT_ROOT
chmod +x train_2_gpu.sh && ./train_2_gpu.sh
```

**Important note:** Our training code currently supports the baseline released with the [Ego4D paper](https://arxiv.org/pdf/2110.07058.pdf). For improved training mechanisms and architectures, we recommend using code from prior [challenge winners](https://github.com/facebookresearch/vq2d_cvpr).

## Evaluating models on VQ2D

We split the evaluation into two steps: (1) Extracting per-frame bbox proposals and estimating their similarity to the visual query, and (2) Peak detection and bidirectional tracking to infer the response track. There are two key benefits to this separation:

* **Rapid hyperparameter searches for step (2):** Step (1) is the most expensive operation as it takes ~24 hours on an 8-GPU + 80-core machine. Once the detections are pre-computed, step (2) only takes ~1-2 hours on the same machine. This allowed us to release improved hyperparameters for step (2) and obtain much better results.
* **Decoupling detector model from our inference code for step (2):** While we support only training the baseline model from the [Ego4D paper](https://arxiv.org/pdf/2110.07058.pdf), we can support inference with arbitrary models as long as the pre-extracted detection scores are available.

**Step (1)** Extracting per-frame bbox proposals.
```
# Note: MODEL_ROOT and DETECTIONS_SAVE_ROOT must be absolute paths
MODEL_ROOT=<path to trained model>  # contains model.pth and config.yaml
DETECTIONS_SAVE_ROOT=<path to save pre-computed detections>

cd $VQ2D_ROOT

# Extract per-frame bbox proposals and visual query similarity scores
chmod +x ./scripts/extract_vq_detections.sh
./scripts/extract_vq_detections.sh val $MODEL_ROOT $DETECTIONS_SAVE_ROOT
./scripts/extract_vq_detections.sh test_unannotated $MODEL_ROOT $DETECTIONS_SAVE_ROOT
```

**Step (2)** Peak detection and bidirectional tracking.

```
./scripts/infer_vq.sh $MODEL_ROOT $DETECTIONS_SAVE_ROOT val 8 0.50 0.10
./scripts/infer_vq.sh $MODEL_ROOT $DETECTIONS_SAVE_ROOT test_unannotated 8 0.50 0.10
```

**Notes:**
* To reduce GPU / CPU usage, reduce 8 from step (2) based on your specific system.

* To get VQ2D evaluation results:
    ```
    python evaluate_vq.py --gt-file data/vq_val.json --pred-file <path to inference json>
    ```
* To participate in the challenge, submit the inference json obtained for the test_unannotated split on evalai.

## Pre-trained models and detection scores
For reproducibility and conveneice,  we provide pre-trained models and corresponding detection scores for the [SiamRCNN baseline](https://arxiv.org/pdf/2110.07058.pdf) and [ImprovedBaselines model](https://github.com/facebookresearch/vq2d_cvpr). They can be downloaded using the ego4d CLI as follows:

```
python -m ego4d.cli.cli -y --output_directory /path/to/output/ --datasets vq2d_models vq2d_detections
```

## Acknowledgements
This codebase relies on [detectron2](https://github.com/facebookresearch/detectron2), [PyTracking](https://github.com/visionml/pytracking), [pfilter](https://github.com/johnhw/pfilter) and [ActivityNet](https://github.com/activitynet/ActivityNet) repositories.

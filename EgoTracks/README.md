# EgoTracks 

## Install packages
```sh
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install EgoTracks as package:
```
python setup.py install
```

## Download clips
Please follow [Start here](https://github.com/EGO4D/docs/blob/main/docs/start-here.md) for instructions on how to access the dataset by accepting the terms of our license agreement.
And then use the following command to download clips and annotations for EgoTracks:
```
ego4d --output_directory ./ --datasets egotracks clips --benchmark EM --version v2
```

## Preprocess - extract frames from exported clips
Replace the following paths for reading and outputing to the correct place in ExtractFramesWorkflowParams in tools/preprocess/extract_ego4d_clip_frames.py: 
* clip_dir: path to the downloaded clip directory
* annotation_path: path to the annotation file we would like to extract frames. If you want the test set only, only extract frames for the test set. If you also would like training part, then need to extract for training set as well.
* output_dir: path to the directory where to the save the extracted frame images. 

And then extract frames from video clips:
```sh
python tools/preprocess/extract_ego4d_clip_frames.py 
```

## Finetuning on EgoTracks
We used STARK (Res50) as pre-trained models, so download the model weights from https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN
Or you can use our trained checkpoint at https://drive.google.com/file/d/14vZmWxYSGJXZGxD5U1LthvvTR_eRzWCw/view?usp=share_link

Change the following paths in the tracking/config/stark_defaults.py:
* cfg.DATA.EGO4DLTT_ANNOTATION_PATH = "your_path/train_v1.json"
* cfg.DATA.EGO4DLTT_DATA_DIR = "your_path_to_extract_frames" - This is the same as the output_dir in preprocess

And change the model weights path and output directory in train.sh, and then run:
```sh
bash train.sh
```


## Infer challenge set and submit challenge result
We use "{clip_uid}\_{query_set_id}\_{object_title}" as unique name for each sequence (object). One could use the EGO4DLTTrackingDataset from tracking/dataset/eval_datasets/ego4d_lt_tracking_dataset.py for loading images and sequence name.
An example of how to run test and generate submission file is in tools/eval_datasets/eval_ego4d_lt_tracking.py and tools/train_net.py result2submission.

Change the following paths in the tracking/config/stark_defaults.py:
* cfg.EVAL.EGO4DLT.ANNOTATION_PATH = "your_path/challenge_test_v1_unannotated.json"
* cfg.EVAL.EGO4DLT.DATA_DIR = "your_path_to_extract_frames" 

And change the model weights path and output directory in test.sh, and then run:
```sh
bash test.sh
```


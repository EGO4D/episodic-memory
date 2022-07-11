
## Environment Installation
Create a conda environment and install required packages from scratch following the steps below
```
    conda create -n pytorch160 python=3.7 
    conda activate pytorch160   
    conda install pytorch=1.6.0 torchvision cudatoolkit=10.1.243 -c pytorch   
    conda install -c anaconda pandas    
    conda install -c anaconda h5py  
    conda install -c anaconda scipy 
    conda install -c conda-forge tensorboardx   
    conda install -c anaconda joblib    
    conda install -c conda-forge matplotlib 
    conda install -c conda-forge urllib3
```
### Annotation conversion 
If you use the canonical annotation files, you need to first convert them by removing unused 
categories and video clips
```
    python Convert_annotation.py
```

### Training
```    
     python Train.py --use_xGPN --is_train true --dataset ego4d --feature_path {DATA_PATH} --checkpoint_path {CHECKPOINT_PATH} --batch_size 32 --train_lr 0.0001
```
### Inference
```
     python Infer.py  --use_xGPN --is_train false --dataset ego4d --feature_path {DATA_PATH} --checkpoint_path {CHECKPOINT_PATH}  --output_path {OUTPUT_PATH}   
```
### Evaluation
```
     python Eval.py --dataset ego4d --output_path {OUTPUT_PATH} --out_prop_map {OUT_PMAP} --eval_stage all
```
### Generate a submission file for the Ego4D Moment Queries challenge
```
    python Merge_detection_retrieval.py
```

## Acknowledgements

This codebase is built on  [VSGN](https://github.com/coolbay/VSGN).

Please also consider citing [VSGN](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Video_Self-Stitching_Graph_Network_for_Temporal_Action_Localization_ICCV_2021_paper.pdf) if you use this codebase.
```
@inproceedings{zhao2021video,
  title={Video self-stitching graph network for temporal action localization},
  author={Zhao, Chen and Thabet, Ali K and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13658--13667},
  year={2021}
}
```

# Span-based Localizing Network for Natural Language Video Localization

This repository adapts the *PyTorch* implementation of the **VSLNet** baseline for the Ego4D: Natural Language
Queries (NLQ) task ([ArXiv][arxiv_link], [webpage][ego4d_page]).
The model is based on the paper "Span-based Localizing Network for Natural Language Video 
Localization" (ACL 2020, long paper, [ACL](vslnet_acl), 
[ArXiv][vslnet_arxiv], [code][vslnet_code]).


## Prerequisites

This repository is based off [VSLNet][vslnet_code]. This has additional support for SLURM scheduling.

### Environment setup

In a pre-existing conda environment or pyenv:
```
pip install -r requirements.txt
```

### Data

```
python3.9 -m nltk.downloader punkt
```

## Preparation

### Dataset 

Download the dataset from the [official webpage][ego4d_page] and place them
in `data/` folder.

Run the preprocessing script using:

```bash
python utils/prepare_ego4d_dataset.py \
    --input_train_split data/nlq_train.json \
    --input_val_split data/nlq_val.json \
    --input_test_split data/nlq_test_unannotated.json \
    --video_feature_read_path data/features/nlq_official_v1/video_features/ \
    --clip_feature_save_path data/features/nlq_official_v1/official \
    --output_save_path data/dataset/nlq_official_v1
```

This creates JSON files in `data/dataset/nlq_official_v1` that can be used for training and evaluating the VSLNet baseline model.


### Video features

Download the official video features released from [official webpage][ego4d_page] and place them in `data/features/nlq_official_v1/video_features/` folder.


## Quick Start

**Train** and **Test**

```shell
# To train the model.
python main.py \
    --task nlq_official_v1 \
    --predictor bert \
    --dim 128 \
    --mode train \
    --video_feature_dim 2304 \
    --max_pos_len 128 \
    --epochs 200 \
    --fv official \
    --num_workers 64 \
    --model_dir checkpoints/ \
    --eval_gt_json "data/nlq_val.json" \
    --log_to_tensoboard "baseline"


# To predict on test set.
python main.py \
    --task nlq_official_v1 \
    --predictor bert \
    --mode test \
    --video_feature_dim 2304 \
    --max_pos_len 128 \
    --fv official \
    --model_dir checkpoints/


# To evaluate predictions using official evaluation script.
PRED_FILE="checkpoints/vslnet_nlq_official_v1_official_512_bert/model"
python utils/evaluate_ego4d_nlq.py \
    --ground_truth_json data/nlq_val.json \
    --model_prediction_json ${PRED_FILE}/vslnet_0_357_preds.json \
    --thresholds 0.3 0.5 \
    --topK 1 3 5
```

### SLURM

We support scheduling with a SLURM cluster with the [submit][submitit_library]
library.  Please provide the flags: `--slurm`, `--slurm_partition <str>` and
`--slurm_timeout_min <int>` to schedule with SLURM. Please provide these flags
to `main.py`.

### Filter 0s Queries

There are queries that are of duration 0s. This is an artifact of the
annotation data. This affects ~14% of the data. You can filter these out via
appending `--remove_empty_queries_from` to `main.py`.

### Tensorboard

Tensorboard will log to loss and validation metrics to `--tb_log_dir`. Losses
will be logged every `--tb_log_freq`, which defaults to every iteration. You can specify the name of the log (for multiple logs) with
`--log_to_tensorboard`.

By default, tensorboard will not be logged unless a parameter to `--log_to_tensorboard` is provided.

## Results

See the main NLQ [README.md][nlq_readme] to compare the performance of all available baselines.


## Citation

If you use the natural language queries (NLQ) dataset, please cite our Ego4D work:

```
@article{Ego4D2021,
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and Martin, Miguel and Nagarajan, Tushar and Radosavovic, Ilija and Ramakrishnan, Santhosh Kumar and Ryan, Fiona and Sharma, Jayant and Wray, Michael and Xu, Mengmeng and Xu, Eric Zhongcong and Zhao, Chen and Bansal, Siddhant and Batra, Dhruv and Cartillier, Vincent and Crane, Sean and Do, Tien and Doulaty, Morrie and Erapalli, Akshay and Feichtenhofer, Christoph and Fragomeni, Adriano and Fu, Qichen and Fuegen, Christian and Gebreselasie, Abrham and Gonzalez, Cristina and Hillis, James and Huang, Xuhua and Huang, Yifei and Jia, Wenqi and Khoo, Weslie and Kolar, Jachym and Kottur, Satwik and Kumar, Anurag and Landini, Federico and Li, Chao and Li, Yanghao and Li, Zhenqiang and Mangalam, Karttikeya and Modhugu, Raghava and Munro, Jonathan and Murrell, Tullie and Nishiyasu, Takumi and Price, Will and Puentes, Paola Ruiz and Ramazanova, Merey and Sari, Leda and Somasundaram, Kiran and Southerland, Audrey and Sugano, Yusuke and Tao, Ruijie and Vo, Minh and Wang, Yuchen and Wu, Xindi and Yagi, Takuma and Zhu, Yunyi and Arbelaez, Pablo and Crandall, David and Damen, Dima and Farinella, Giovanni Maria and Ghanem, Bernard and Ithapu, Vamsi Krishna and Jawahar, C. V. and Joo, Hanbyul and Kitani, Kris and Li, Haizhou and Newcombe, Richard and Oliva, Aude and Park, Hyun Soo and Rehg, James M. and Sato, Yoichi and Shi, Jianbo and Shou, Mike Zheng and Torralba, Antonio and Torresani, Lorenzo and Yan, Mingfei and Malik, Jitendra},
  title     = {Ego4D: Around the {W}orld in 3,000 {H}ours of {E}gocentric {V}ideo},
  journal   = {CoRR},
  volume    = {abs/2110.07058},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07058},
  eprinttype = {arXiv},
  eprint    = {2110.07058}
}
```

If you are using this baseline, please also cite the original work:

```
@inproceedings{zhang2020span,
    title = "Span-based Localizing Network for Natural Language Video Localization",
    author = "Zhang, Hao  and Sun, Aixin  and Jing, Wei  and Zhou, Joey Tianyi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.585",
    pages = "6543--6554"
}
```

[arxiv_link]:https://arxiv.org/abs/2110.07058 
[ego4d_page]: https://ego4d-data.org/
[vslnet_arxiv]: https://arxiv.org/abs/2004.13931
[vslnet_acl]: https://www.aclweb.org/anthology/2020.acl-main.585.pdf
[vslnet_code]: https://github.com/IsaacChanghau/VSLNet
[nlq_readme]:./../README.md
[submitit_library]: https://github.com/facebookincubator/submitit

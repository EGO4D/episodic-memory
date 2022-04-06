
## Environment Installation
Please refer to README\_2D-TAN.MD for the main environment.

After that, please install transformers package
```
pip install transformers
```


### Train && Test

To train a new model
```    
python moment_localization/train.py --cfg  experiments/ego4d/2D-TAN-40x40-K9L4-pool-window-std-sf.yaml --verbose
```

To generate the submission file, do
```
python moment_localization/test.py --cfg  experiments/ego4d/2D-TAN-40x40-K9L4-pool-window-std-sf.yaml --verbose --split test --result nlq_2dtan_submission.json
```

## Acknowledgements

This codebase is built on  [2D-TAN](https://github.com/microsoft/2D-TAN).

## Citation

If any part of the code is helpful to your work, please cite with:

```
@InProceedings{
2DTAN_2020_AAAI,
author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
booktitle = {AAAI},
year = {2020}
}
```

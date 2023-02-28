python tools/train_net.py \
--num-gpus 1 \
MODEL.WEIGHTS /checkpoint/haotang/experiments/EgoTracks/STARKST_ep0001.pth.tar \
\
OUTPUT_DIR /checkpoint/haotang/experiments/EgoTracks/res/stark_finetune_Ego4DLTT \
TRAIN_STAGE_1.EPOCH 10 \
TRAIN_STAGE_1.LR_DROP_EPOCH 8 \
TRAIN_STAGE_1.NUM_WORKER 2 \
TRAIN_STAGE_2.EPOCH 1 \
TRAIN_STAGE_2.NUM_WORKER 1 \
DATA.TRAIN.DATASETS_NAME "[\"EGO4DLTT\"]" \
DATA.TRAIN.DATASETS_RATIO "[1]" \

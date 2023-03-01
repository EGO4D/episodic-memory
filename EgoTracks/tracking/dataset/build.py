import os
from typing import List

import tracking.dataset.transforms as tfm
from torch.utils.data.distributed import DistributedSampler

# datasets related
from tracking.dataset.dataloader import LTRLoader
from tracking.dataset.processing.stark_processing import STARKProcessing
from tracking.dataset.trackingdataset import TrackingDataset
from tracking.dataset.train_datasets.coco_seq import MSCOCOSeq
from tracking.dataset.train_datasets.ego4d_vq import Ego4DVQ
from tracking.dataset.train_datasets.got10k import Got10k
from tracking.dataset.train_datasets.ego4d_lt_tracking import EGO4DLTT
from tracking.dataset.train_datasets.lasot import Lasot
from tracking.dataset.train_datasets.tracking_net import TrackingNet
from tracking.utils.utils import opencv_loader


# TODO: Add other dataset class
def names2datasets(name_list: List, cfg, image_loader=opencv_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in [
            "LASOT",
            "GOT10K_vottrain",
            "GOT10K_votval",
            "GOT10K_train_full",
            "COCO17",
            "VID",
            "TRACKINGNET",
            "EGO4DVQ",
            "EGO4DLTT",
        ], f"Dataset {name} not found!"
        if name == "LASOT":
            datasets.append(
                Lasot(
                    cfg.DATA.LASOT_DATA_DIR,
                    split="train",
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                )
            )
        if name == "GOT10K_vottrain":
            datasets.append(
                Got10k(
                    os.path.join(cfg.DATA.GOT10K_DATA_DIR, "train"),
                    split="vottrain",
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                    cached_sequence_meta_info_dir=cfg.DATA.CACHED_GOT10K_META_INFO_DIR,
                )
            )
        if name == "GOT10K_train_full":
            datasets.append(
                Got10k(
                    os.path.join(cfg.DATA.GOT10K_DATA_DIR, "train"),
                    split="train_full",
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                    cached_sequence_meta_info_dir=cfg.DATA.CACHED_GOT10K_META_INFO_DIR,
                )
            )
        if name == "GOT10K_votval":
            datasets.append(
                Got10k(
                    os.path.join(cfg.DATA.GOT10K_DATA_DIR, "train"),
                    split="votval",
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                    cached_sequence_meta_info_dir=cfg.DATA.CACHED_GOT10K_META_INFO_DIR,
                )
            )
        if name == "COCO17":
            datasets.append(
                MSCOCOSeq(
                    cfg.DATA.COCO_DATA_DIR,
                    version="2017",
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                )
            )
        if name == "EGO4DVQ":
            datasets.append(
                Ego4DVQ(
                    cfg.DATA.EGO4DVQ_DATA_DIR,
                    cfg.DATA.EGO4DVQ_ANNOTATION_PATH,
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                )
            )
        # if name == "VID":
        #     datasets.append(
        #         ImagenetVID(
        #             cfg.DATA.VID_DATA_DIR,
        #             image_loader=image_loader,
        #             data_fraction=cfg.DATA.DATA_FRACTION,
        #         )
        #     )
        if name == "TRACKINGNET":
            datasets.append(
                TrackingNet(
                    cfg.DATA.TRACKINGNET_DATA_DIR,
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                    cached_sequence_list_dir=cfg.DATA.CACHED_TRACKINGNET_SEQUENCE_LIST_DIR,
                )
            )

        if name == "EGO4DLTT":
            datasets.append(
                EGO4DLTT(
                    cfg.DATA.EGO4DLTT_DATA_DIR,
                    cfg.DATA.EGO4DLTT_ANNOTATION_PATH,
                    image_loader=image_loader,
                    data_fraction=cfg.DATA.DATA_FRACTION,
                )
            )

    return datasets


def build_dataloaders(cfg, local_rank=-1):
    # Data transform
    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5)
    )

    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.RandomHorizontalFlip_Norm(probability=0.5),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    )

    transform_val = tfm.Transform(
        tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )

    # The tracking pairs processing module
    output_sz = {"template": cfg.DATA.TEMPLATE.SIZE, "search": cfg.DATA.SEARCH.SIZE}
    search_area_factor = {
        "template": cfg.DATA.TEMPLATE.FACTOR,
        "search": cfg.DATA.SEARCH.FACTOR,
    }
    center_jitter_factor = {
        "template": cfg.DATA.TEMPLATE.CENTER_JITTER,
        "search": cfg.DATA.SEARCH.CENTER_JITTER,
    }
    scale_jitter_factor = {
        "template": cfg.DATA.TEMPLATE.SCALE_JITTER,
        "search": cfg.DATA.SEARCH.SCALE_JITTER,
    }

    data_processing_train = STARKProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=center_jitter_factor,
        scale_jitter_factor=scale_jitter_factor,
        mode="sequence",
        transform=transform_train,
        joint_transform=transform_joint,
    )

    data_processing_val = STARKProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=center_jitter_factor,
        scale_jitter_factor=scale_jitter_factor,
        mode="sequence",
        transform=transform_val,
        joint_transform=transform_joint,
    )

    # Train sampler and loader
    num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = TrackingDataset(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, cfg, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=num_search,
        num_template_frames=num_template,
        processing=data_processing_train,
        frame_sample_mode=sampler_mode,
        train_cls=train_cls,
    )

    train_sampler = DistributedSampler(dataset_train) if local_rank != -1 else None
    shuffle = False if local_rank != -1 else True

    loader_train = LTRLoader(
        "train",
        dataset_train,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=train_sampler,
    )

    # Validation samplers and loaders
    dataset_val = TrackingDataset(
        datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, cfg, opencv_loader),
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=num_search,
        num_template_frames=num_template,
        processing=data_processing_val,
        frame_sample_mode=sampler_mode,
        train_cls=train_cls,
    )
    val_sampler = DistributedSampler(dataset_val) if local_rank != -1 else None
    loader_val = LTRLoader(
        "val",
        dataset_val,
        training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=val_sampler,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL,
    )

    return loader_train, loader_val

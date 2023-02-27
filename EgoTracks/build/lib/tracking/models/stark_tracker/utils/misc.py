# lib.utils.misc

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    # TODO: this is to disable download resnet50 checkpoint.
    # change the url to the right manifold location
    return get_rank() == 0


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))

    def process(
        self,
        img_arr: np.ndarray,
        amask_arr: np.ndarray,
        device: torch.device = None,
        verbose: bool = False,
    ):
        # Deal with the image patch
        if device is not None:
            img_tensor = (
                torch.tensor(img_arr)
                .float()
                .permute((2, 0, 1))
                .unsqueeze(dim=0)
                .to(device)
            )
            t = time.time()
            img_tensor_norm = (
                (img_tensor / 255.0) - self.mean.to(device)
            ) / self.std.to(
                device
            )  # (1,3,H,W)

            if verbose:
                logging.error(f"Normalize image {time.time() - t}")

            # Deal with the attention mask
            amask_tensor = (
                torch.from_numpy(amask_arr).to(torch.bool).unsqueeze(dim=0)
            ).to(
                device
            )  # (1,H,W)
        else:
            img_tensor = (
                torch.tensor(img_arr).float().permute((2, 0, 1)).unsqueeze(dim=0)
            )
            img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
            # Deal with the attention mask
            amask_tensor = (
                torch.from_numpy(amask_arr).to(torch.bool).unsqueeze(dim=0)
            )  # (1,H,W)

        return NestedTensor(img_tensor_norm, amask_tensor)

from typing import List, Any

import torch
from torch.nn import functional as F


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *args: Any,
    reduction: str = "mean",
    flip_class: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Same as `torch.nn.functional.binary_cross_entropy`, but returns 0 (instead of nan)
    for empty inputs. `flip_class` inverts the target since the positive class is 0 and the
    negative/background class is 1.
    """

    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    target = target.float()
    if flip_class:
        target = 1 - target
    return F.binary_cross_entropy(input, target, reduction=reduction, **kwargs)


def binary_cross_entropy_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    *args: Any,
    reduction: str = "mean",
    flip_class: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Same as `torch.nn.functional.binary_cross_entropy_with_logits`, but returns 0 (instead of nan)
    for empty inputs. `flip_class` inverts the target since the positive class is 0 and the
    negative/background class is 1.
    """

    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    target = target.float()
    if flip_class:
        target = 1 - target
    return F.binary_cross_entropy_with_logits(
        input, target, reduction=reduction, **kwargs
    )


def kl_div(
    input: torch.Tensor,
    target: torch.Tensor,
    *args: Any,
    reduction: str = "mean",
    flip_class: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Same as `torch.nn.functional.kl_div`, but returns 0 (instead of nan)
    for empty inputs. `flip_class` inverts the target since the positive class is 0 and the
    negative/background class is 1.
    """

    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    target = target.float()
    if flip_class:
        target = 1 - target
    # convert input from logits to log_probs
    input = torch.log_softmax(input, dim=1)
    # normalize target to probabilities
    target = target / (target.sum(dim=1, keepdim=True) + 1e-10)
    return F.kl_div(input, target, reduction=reduction, **kwargs)


def triplet_margin(
    input: torch.Tensor,
    target: torch.Tensor,
    *args: Any,
    reduction: str = "mean",
    flip_class: bool = True,
    margin: float = 0.25,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Returns 0 (instead of nan) for empty inputs. `flip_class` inverts the target
    since the positive class is 0 and the negative/background class is 1.
    Args:
        input - (N, 1) normalized similarity values
        target - (N, ) target classes. By default, positive is 0, negative is 1.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    target = target.float()
    if flip_class:
        target = 1 - target
    positive_idxs = torch.where(target == 1)[0]
    negative_idxs = torch.where(target == 0)[0]
    loss = 0.0
    count = 0.0
    for p in positive_idxs.detach().cpu().tolist():
        for n in negative_idxs.detach().cpu().tolist():
            dp = (1 - input[p, 0]) / 2.0
            dn = (1 - input[n, 0]) / 2.0
            loss = loss + torch.clamp(dp - dn + margin, 0)
            count = count + 1
    loss = loss / count
    return loss

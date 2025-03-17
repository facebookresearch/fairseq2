# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.nn.functional import nll_loss as _nll_loss


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_idx: int | None,
    *,
    label_smoothing: float = 0.0,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> Tensor:
    """Computes the cross entropy loss between ``logits`` and ``targets``.

    This function differs from :func:`torch.nn.functional.cross_entropy` in
    two ways:
        - It uses a fused kernel based on ``log_sofmax()`` and ``nll_loss()``
          which typically consumes less memory while having the same performance.
        - Its label smoothing implementation is slightly different and has
          parity with the original fairseq implementation.

    :param logits: The logits. *Shape:* :math:`(S,T)`, where :math:`S` is the
        sequence length and :math:`T` is the size of the vocabulary.
    :param targets: The target indices. *Shape:* :math:`(S)`, where :math:`S` is
        the sequence length.
    :param pad_idx: The index of the PAD symbol in the target vocabulary.
    :param label_smoothing: The amount of label smoothing to apply while
        computing the loss.
    :param reduction: The reduction to apply to the output.
    """
    # If we don't require label smoothing, use the fused cross entropy with
    # the `nll_loss()` of PyTorch as it is faster than our implementation.
    if label_smoothing == 0.0:
        if pad_idx is None:
            pad_idx = -100

        return _fused_cross_entropy(logits, targets, pad_idx, reduction)

    lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

    # sum/mean: (), none: (S)
    return nll_loss(
        lprobs, targets, pad_idx, label_smoothing=label_smoothing, reduction=reduction
    )


@torch.compile(fullgraph=True)
def _fused_cross_entropy(
    logits: Tensor, targets: Tensor, pad_idx: int, reduction: str
) -> Tensor:
    # For numerical stability run in single precision.
    # (S, T) -> (S, T)
    lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

    return _nll_loss(lprobs, targets, ignore_index=pad_idx, reduction=reduction)


def nll_loss(
    lprobs: Tensor,
    targets: Tensor,
    pad_idx: int | None,
    *,
    label_smoothing: float = 0.0,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> Tensor:
    """Computes the negative log-likelihood loss.

    In addition to :func:`torch.nn.functional.nll_loss`, this function accepts
    a loss smoothing parameter that is compatible with the original fairseq.

    :param lprobs: The log probabilities. *Shape:* :math:`(S,T)`, where :math:`S`
        is the sequence length and :math:`T` is the size of the vocabulary.
    :param targets: The target indices. *Shape:* :math:`(S)`, where :math:`S` is
        the sequence length.
    :param pad_idx: The index of the PAD symbol in the target vocabulary.
    :param label_smoothing: The amount of label smoothing to apply while
        computing the loss.
    :param reduction: The reduction to apply to the output.
    """
    # (S) -> (S, 1)
    targets = targets.unsqueeze(-1)

    loss = -lprobs.gather(dim=-1, index=targets)

    if label_smoothing > 0.0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    else:
        smooth_loss = None

    if pad_idx is not None:
        padding_mask = targets.eq(pad_idx)

        loss.masked_fill_(padding_mask, 0.0)

        if smooth_loss is not None:
            smooth_loss.masked_fill_(padding_mask, 0.0)

    if reduction == "sum":
        loss = loss.sum()

        if smooth_loss is not None:
            smooth_loss = smooth_loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

        if smooth_loss is not None:
            smooth_loss = smooth_loss.mean()

    if smooth_loss is not None:
        # Our label smoothing implementation varies slightly from PyTorch's
        # implementation in `cross_entropy`.
        eps = label_smoothing / (lprobs.size(-1) - 1)

        loss = (1.0 - label_smoothing - eps) * loss + eps * smooth_loss

    if reduction == "none":
        loss = loss.squeeze(-1)

    return loss

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.nn.functional import nll_loss as torch_nll_loss


class CrossEntropy(Protocol):
    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        pad_idx: int | None,
        *,
        label_smoothing: float = 0.0,
        reduction: Literal["sum", "mean", "none"] = "sum",
    ) -> Tensor: ...


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_idx: int | None,
    *,
    label_smoothing: float = 0.0,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> Tensor:
    """
    Computes the cross entropy loss between ``logits`` and ``targets``.

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
    # For numerical stability run in single precision.
    # (S, T) -> (S, T)
    lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

    if label_smoothing == 0.0:
        if pad_idx is None:
            pad_idx = -100

        # Unless we need label smoothing, use PyTorch `nll_loss()` as it can be
        # fused with `log_softmax()` when compiled.
        return torch_nll_loss(
            lprobs, targets, ignore_index=pad_idx, reduction=reduction
        )

    # sum/mean: (), none: (S)
    return nll_loss(
        lprobs, targets, pad_idx, label_smoothing=label_smoothing, reduction=reduction
    )


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


def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """Repeat elements of a tensor.

    :param x:
        The input tensor.
    :param dim:
        The dimension along which to repeat values.
    :param repeat:
        The number of repetitions.

    :returns:
        The repeated tensor which has the same shape as input, except along the
        given axis.

    .. note::
        This is a lightweight version of :func:`torch.repeat_interleave` that
        is faster for repetitions along a single dimension.
    """
    if repeat == 1:
        return x

    shape = [-1] * (x.ndim + 1)

    if dim < 0:
        dim += x.ndim

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)

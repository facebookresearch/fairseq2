# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import log_softmax, nll_loss


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_idx: int | None,
    *,
    label_smoothing: float = 0.0,
    target_mask: Tensor | None = None,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> Tensor:
    """
    Computes the cross entropy loss.

    .. note::
        The loss smoothing implementation of this function is compatible with
        fairseq.
    """
    if logits.ndim == 3:
        batch_size = logits.size(0)

        # (N, S, T) -> (N x S, T)
        logits = logits.flatten(0, 1)
    else:
        batch_size = None

    if targets.ndim == 2:
        # (N, S) -> (N x S)
        targets = targets.flatten()

    # For numerical stability run in single precision.
    # (S, T) -> (S, T)
    log_probs = log_softmax(logits, dim=-1, dtype=torch.float32)

    if label_smoothing == 0.0:
        if pad_idx is None:
            pad_idx = -100

        # (S)
        loss = nll_loss(
            log_probs,
            targets,
            ignore_index=pad_idx,
            reduction=reduction if target_mask is None else "none",
        )

        if target_mask is None:
            if reduction == "none" and batch_size is not None:
                # (N x S) -> (N, S)
                loss = loss.unflatten(0, (batch_size, -1))

            return loss

        if target_mask.ndim == 2:
            # (N, S) -> (N x S)
            target_mask = target_mask.flatten(0, 1)

        # (S)
        loss = loss * target_mask

        if reduction == "sum":
            return loss.sum()

        if reduction == "mean":
            return loss.mean()

        if reduction == "none":
            if batch_size is not None:
                # (N x S) -> (N, S)
                loss = loss.unflatten(0, (batch_size, -1))

            return loss

        raise ValueError(
            f"`reduction` must be 'sum', 'mean' or 'none', but is '{reduction}' instead."
        )

    # (S) -> (S, 1)
    targets = targets.unsqueeze(-1)

    # (S, 1)
    loss = -log_probs.gather(dim=-1, index=targets)

    # (S, 1)
    if label_smoothing > 0.0:
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)
    else:
        smooth_loss = None

    if pad_idx is not None:
        padding_mask = targets.eq(pad_idx)

        loss.masked_fill_(padding_mask, 0.0)

        if smooth_loss is not None:
            smooth_loss.masked_fill_(padding_mask, 0.0)

    if target_mask is not None:
        if target_mask.ndim == 2:
            # (N, S) -> (N x S)
            target_mask = target_mask.flatten(0, 1)

        # (S) -> (S, 1)
        target_mask = target_mask.unsqueeze(-1)

        # (S, 1)
        loss = loss * target_mask

        # (S, 1)
        if smooth_loss is not None:
            smooth_loss = smooth_loss * target_mask

    if reduction == "sum":
        # ()
        loss = loss.sum()

        # ()
        if smooth_loss is not None:
            smooth_loss = smooth_loss.sum()
    elif reduction == "mean":
        # ()
        loss = loss.mean()

        # ()
        if smooth_loss is not None:
            smooth_loss = smooth_loss.mean()
    elif reduction != "none":
        raise ValueError(
            f"`reduction` must be 'sum', 'mean' or 'none', but is '{reduction}' instead."
        )

    if smooth_loss is not None:
        # This label smoothing implementation is identical to the one in fairseq
        # and varies slightly from PyTorch's version in `cross_entropy`.
        eps = label_smoothing / (log_probs.size(-1) - 1)

        loss = ((1.0 - label_smoothing - eps) * loss) + (eps * smooth_loss)

    if reduction == "none":
        # (S, 1) -> (S)
        loss = loss.squeeze(-1)

        if batch_size is not None:
            # (N x S) -> (N, S)
            loss = loss.unflatten(0, (batch_size, -1))

    return loss

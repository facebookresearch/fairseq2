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
from torch.nn.functional import nll_loss as torch_nll_loss

from fairseq2.nn.functional._nll_loss import nll_loss


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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Optional

from torch import Tensor


def nll_loss(
    lprobs: Tensor,
    targets: Tensor,
    pad_idx: Optional[int],
    *,
    label_smoothing: float = 0.0,
    reduction: Literal["none", "sum"] = "sum",
) -> Tensor:
    """Compute the negative log-likelihood loss.

    In contrast to :func:`torch.nn.functional.nll_loss`, this function expects
    ``lprobs`` to be of shape :math:`(N,S,T)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`T` is the size of the
    target vocabulary. The loss is computed over the last dimension which avoids
    strided access and improves runtime performance, in particular for large
    vocabularies.

    :param lprobs:
        The log probabilities. *Shape:* See the function description.
    :param targets:
        The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
        size and :math:`S` is the sequence length.
    :param pad_idx:
        The index of the PAD symbol in the target vocabulary.
    :param label_smoothing:
        The amount of label smoothing to apply while computing the loss.
    :param reduction:
        The reduction to apply to the output.
    """
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

    if smooth_loss is not None:
        # Our label smoothing implementation varies slightly from PyTorch's
        # implementation in `cross_entropy`.
        eps = label_smoothing / (lprobs.size(-1) - 1)

        loss = (1.0 - label_smoothing - eps) * loss + eps * smooth_loss

    if reduction == "none":
        loss = loss.squeeze(-1)

    return loss

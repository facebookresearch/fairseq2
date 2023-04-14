# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
from packaging import version
from torch import Tensor

log = logging.getLogger(__name__)

is_pt2_or_greater = version.parse(torch.__version__) >= version.parse("2.0.0")

has_warned_sdpa = False


class AttentionFunction(Protocol):
    """Computes attention."""

    def __call__(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        needs_weights: bool = False,
        training: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x:
            The input to query. *Shape:* :math:`(N,S,K)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`K` is
            the key size.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(S,S_{kv})` or
            :math:`(N,S,S_{kv})`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`S_{kv}` is the
            key/value sequence length.
        :param dropout_p:
            The dropout probability on the attention weights.
        :param needs_weights:
            A boolean value indicating whether the function should return the
            attention weights. If ``True``, the second item of the returned
            tuple will contain the attention weights.
        :param training:
            If ``True``, applies dropout.

        :returns:
            - The attention values. *Shape:* :math:`(N,S,V)`, where
              :math:`N` is the batch size, :math:`S` is the sequence length, and
              :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`S` is the sequence length, and
              :math:`S_{kv}` is the key/value sequence length.
        """


def default_scaled_dot_product_attention(
    x: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    This function automatically picks the most efficient SDPA implementation and
    uses it internally.

    .. note::
        This function follows the :class:`AttentionFunction` protocol.
    """
    global has_warned_sdpa

    if not needs_weights and is_pt2_or_greater and x.is_cuda:
        return torch_scaled_dot_product_attention(
            x, keys, values, mask, dropout_p, needs_weights, training
        )
    else:
        if is_pt2_or_greater and x.is_cuda and not has_warned_sdpa:
            log.warning(
                "You are failing to leverage the more efficient `torch_scaled_dot_product_attention` that is based on PyTorch's native SDPA because of `needs_weights` set to `True`."
            )

            has_warned_sdpa = True

        return naive_scaled_dot_product_attention(
            x, keys, values, mask, dropout_p, needs_weights, training
        )


def torch_scaled_dot_product_attention(
    x: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute scaled dot-product attention using PyTorch SDPA 2.0 as described
    `here <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
    in the PyTorch documentation.

    .. note::
        This function follows the :class:`AttentionFunction` protocol.
    """
    if not is_pt2_or_greater:
        raise ValueError(
            "`torch_scaled_dot_product_attention` requires PyTorch 2.0.0 or greater."
        )

    if needs_weights:
        raise ValueError(
            "`torch_scaled_dot_product_attention` does not support the `needs_weights` parameter."
        )

    if not training:
        dropout_p = 0.0

    # Check if the mask is tagged as causal.
    is_causal: bool = getattr(mask, "is_causal", False)

    attn = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
        x,
        keys,
        values,
        attn_mask=None if is_causal else mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

    return attn, None


def naive_scaled_dot_product_attention(
    x: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute scaled dot-product attention using a non-fused Python-based
    implementation.

    This function is used internally by :func:`default_scaled_dot_product_attention`
    as a fall-back if no other efficient implementation is available.

    It can also be used explicitly for troubleshooting purposes since it allows
    step-through debugging.

    .. note::
        This function follows the :class:`AttentionFunction` protocol.
    """
    x = x * (x.size(-1) ** -0.5)

    if mask is None:
        # (N, S, K) @ (N, K, S_kv) = (N, S, S_kv)
        attn_weights = torch.bmm(x, keys.transpose(1, 2))
    else:
        # (N, S, S_kv) + ((N, S, K) @ (N, K, S_kv)) = (N, S, S_kv)
        attn_weights = torch.baddbmm(mask, x, keys.transpose(1, 2))

    attn_weights = F.softmax(attn_weights, dim=-1)

    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, dropout_p, training)

    # (N, S, S_kv) @ (N, S_kv, V) = (N, S, V)
    attn = torch.bmm(attn_weights, values)

    return attn, attn_weights if needs_weights else None

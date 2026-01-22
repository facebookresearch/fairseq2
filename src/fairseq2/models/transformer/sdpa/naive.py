# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from torch.nn.functional import dropout, softmax
from typing_extensions import override

from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout


@final
class NaiveSDPA(SDPA):
    """Computes scaled dot-product attention using a Python implementation."""

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # ([[N], H], S, S_kv)
        bias = maybe_get_attention_bias_tensor(
            self.bias, q, q_layout, k_layout, bias_cache
        )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        # (N, S, H, K) -> (N, H, S, K)
        q = q.transpose(-2, -3)

        # (N, S_kv, H, K) -> (N, H, S_kv, K)
        k = k.transpose(-2, -3)

        # (N, S_kv, H, V) -> (N, H, S_kv, V)
        v = v.transpose(-2, -3)

        attns, attn_weights = naive_scaled_dot_product_attention(
            q, k, v, bias, dropout_p=dropout_p, needs_weights=needs_weights
        )

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, attn_weights

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"


def naive_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor | None,
    *,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
) -> tuple[Tensor, Tensor | None]:
    # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
    weights = torch.matmul(q, k.transpose(-1, -2))

    weights = weights * (q.size(-1) ** -0.5)

    if bias is not None:
        # (N, H, S, S_kv) + ([[N], H], S, S_kv) -> (N, H, S, S_kv)
        weights = weights + bias

    # For numerical stability run in single precision.
    weights = softmax(weights, dim=-1, dtype=torch.float32)

    weights = weights.type_as(q)

    if dropout_p > 0.0:
        weights = dropout(weights, dropout_p)

    # (N, H, S, S_kv) @ (N, H, S_kv, V) = (N, H, S, V)
    attns = torch.matmul(weights, v)

    return attns, weights if needs_weights else None

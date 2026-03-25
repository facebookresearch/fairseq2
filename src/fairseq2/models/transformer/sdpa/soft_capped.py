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
class SoftCappedSDPA(SDPA):
    """Computes scaled dot-product attention with soft-capped logits.

    Applies soft-capping to attention logits using tanh:
        logits_capped = soft_cap * tanh(logits / soft_cap)

    This prevents attention logits from growing too large, which can improve
    training stability and model quality.
    """

    bias: AttentionBias
    soft_cap: float
    dropout_p: float

    def __init__(
        self,
        bias: AttentionBias,
        *,
        soft_cap: float = 30.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.bias = bias
        self.soft_cap = soft_cap
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

        # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
        weights = torch.matmul(q, k.transpose(-1, -2))

        weights = weights * (q.size(-1) ** -0.5)

        # Apply soft-capping: tanh(logits / cap) * cap
        weights = torch.tanh(weights / self.soft_cap) * self.soft_cap

        if bias is not None:
            # (N, H, S, S_kv) + ([[N], H], S, S_kv) -> (N, H, S, S_kv)
            weights = weights + bias

        # For numerical stability run in single precision
        weights = softmax(weights, dim=-1, dtype=torch.float32)

        weights = weights.type_as(q)

        if dropout_p > 0.0:
            weights = dropout(weights, dropout_p)

        # (N, H, S, S_kv) @ (N, H, S_kv, V) = (N, H, S, V)
        attns = torch.matmul(weights, v)

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, weights if needs_weights else None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"bias={self.bias}, soft_cap={self.soft_cap:G}, "
            f"dropout_p={self.dropout_p:G}"
        )

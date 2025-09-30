# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, final

import torch
from torch import Tensor
from typing_extensions import override

try:
    from flash_attn import (  # type: ignore[import-not-found, import-untyped]
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    _has_flash_attn_2 = False
else:
    _has_flash_attn_2 = True

from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout


@final
class Flash2SDPA(SDPA):
    """Computes scaled dot-product attention using FlashAttention2."""

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
        if not _has_flash_attn_2:
            raise OperationalError(
                "FlashAttention is not found. Use `pip install flash-attn`."
            )

        if q_layout.padded or k_layout.padded:
            raise NotSupportedError(f"`{Flash2SDPA}` does not support padded batches.")

        if isinstance(self.bias, IdentityBias):
            causal = False

            window_size = (-1, -1)
        elif isinstance(self.bias, CausalAttentionBias):
            causal = True

            attn_window_len = self.bias.attn_window_len
            if attn_window_len is not None:
                left_window = attn_window_len
            else:
                left_window = -1

            window_size = (left_window, -1)
        else:
            raise NotSupportedError(f"`{Flash2SDPA}` does not support `{self.bias}`.")

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        if q_layout.packed ^ k_layout.packed:
            raise ValueError("`q_layout` and `k_layout` must be both packed.")

        device_type = q.device.type

        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)

            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)

        if q_layout.packed:
            attns = flash_attn_varlen_func(
                q,
                k,
                v,
                q_layout.seq_begin_indices_pt,
                k_layout.seq_begin_indices_pt,
                q_layout.max_seq_len,
                k_layout.max_seq_len,
                dropout_p,
                causal=causal,
                window_size=window_size,
            )
        else:
            attns = flash_attn_func(
                q, k, v, dropout_p, causal=causal, window_size=window_size
            )

        attns = cast(Tensor, attns)

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"

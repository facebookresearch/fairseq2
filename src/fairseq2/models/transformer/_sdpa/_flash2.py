# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, final

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

from fairseq2.error import NotSupportedError
from fairseq2.nn import BatchLayout

# isort: split

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer._sdpa._base import SDPA


@final
class Flash2SDPA(SDPA):
    """Computes scaled dot-product attention using FlashAttention2."""

    bias: AttentionBias
    dropout_p: float

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias

        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if not _has_flash_attn_2:
            raise RuntimeError(
                "FlashAttention is not found in your Python environment. Use `pip install flash-attn`."
            )

        if seqs_layout.padded or keys_layout.padded:
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

        if seqs_layout.packed ^ keys_layout.packed:
            raise ValueError("`seqs_layout` and `keys_layout` must be both packed.")

        if seqs_layout.packed:
            attns = flash_attn_varlen_func(
                seqs,
                keys,
                values,
                seqs_layout.seq_begin_indices_pt,
                keys_layout.seq_begin_indices_pt,
                seqs_layout.max_seq_len,
                keys_layout.max_seq_len,
                dropout_p,
                causal=causal,
                window_size=window_size,
            )
        else:
            attns = flash_attn_func(
                seqs, keys, values, dropout_p, causal=causal, window_size=window_size
            )

        attns = cast(Tensor, attns)

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"

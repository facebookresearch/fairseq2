# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.nn import BatchLayout


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    bias: AttentionBias

    def __init__(self, bias: AttentionBias) -> None:
        super().__init__()

        self.bias = bias

    @abstractmethod
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
        """
        :param seqs: The sequences to query. *Shape:* :math:`(N,S,H,K)`, where
            :math:`N` is the batch size, :math:`H` is the number of heads,
            :math:`S` is the sequence length, and :math:`K` is the key size.
        :param keys: The keys. *Shape:* :math:`(N,S_{kv},H,K)`, where :math:`N`
            is the batch size, :math:`H` is the number of heads, :math:`S_{kv}`
            is the key/value sequence length, and :math:`K` is the key size.
        :param values: The values. *Shape:* :math:`(N,S_{kv},H,V)`, where
            :math:`N` is the batch size, :math:`H` is the number of heads,
            :math:`S_{kv}` is the key/value sequence length, and :math:`V` is
            the value size.
        :param needs_weights: If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(N,S,H,V)`, where :math:`N`
              is the batch size, :math:`H` is the number of heads, :math:`S` is
              the sequence length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,H,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`H` is the number of heads,
              :math:`S` is the sequence length, and :math:`S_{kv}` is the
              key/value sequence length.
        """

    @final
    @torch.compiler.disable
    def _maybe_get_attention_bias_tensor(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys_layout: BatchLayout,
        bias_cache: AttentionBiasCache,
    ) -> Tensor | None:
        if isinstance(self.bias, IdentityBias):
            full_seqs = not seqs_layout.packed and not seqs_layout.padded
            full_keys = not keys_layout.packed and not keys_layout.padded

            if full_seqs and full_keys:
                return None

        if isinstance(self.bias, CausalAttentionBias):
            if not seqs_layout.packed:
                if seqs_layout.max_seq_len == 1:
                    return None

        impl = "tensor"

        bias = bias_cache.maybe_get(self.bias, impl, kls=Tensor)
        if bias is None:
            bias = self.bias.materialize(
                seqs_layout, keys_layout, seqs.device, seqs.dtype
            )

            bias_cache.set(self.bias, impl, bias)

        return bias

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}"

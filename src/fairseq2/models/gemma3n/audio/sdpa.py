# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.models.transformer.sdpa.naive import (
    naive_scaled_dot_product_attention,
)
from fairseq2.nn import BatchLayout, StandardEmbedding


@final
class Gemma3nConformerSDPA(SDPA):
    """SDPA for Gemma3n audio conformer with Shaw relative positions,
    chunked local attention, per-dimension scaling, and softcapping.
    """

    model_dim: int
    num_heads: int
    head_dim: int
    max_left_rel_pos: int
    max_right_rel_pos: int
    chunk_size: int
    left_context: int
    right_context: int
    logit_cap: float
    rel_k_embed: StandardEmbedding
    per_dim_scale: nn.Parameter

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_pos: int,
        max_right_rel_pos: int,
        chunk_size: int,
        left_context: int,
        right_context: int,
        logit_cap: float,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality (1536).
        :param num_heads: Number of attention heads (8).
        :param max_left_rel_pos: Maximum left relative position for Shaw embeddings.
        :param max_right_rel_pos: Maximum right relative position for Shaw embeddings.
        :param chunk_size: Chunk size for local attention (12).
        :param left_context: Left context size in tokens (13).
        :param right_context: Right context size in tokens (0).
        :param logit_cap: Logit softcapping value (50.0).
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.max_left_rel_pos = max_left_rel_pos
        self.max_right_rel_pos = max_right_rel_pos
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        self.logit_cap = logit_cap

        num_pos = max_left_rel_pos + 1 + max_right_rel_pos

        self.rel_k_embed = StandardEmbedding(
            num_pos, self.head_dim, init_fn=init_shaw_embedding, device=device, dtype=dtype
        )

        self.per_dim_scale = nn.Parameter(
            torch.ones(self.head_dim, device=device, dtype=dtype)
        )

        self._audio_mask: Tensor | None = None

    def set_audio_mask(self, mask: Tensor | None) -> None:
        """Store audio mel mask for validity masking in chunked attention."""
        self._audio_mask = mask

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
        """
        :param q: Queries. *Shape:* :math:`(N,S,H,K)`.
        :param k: Keys. *Shape:* :math:`(N,S,H,K)`.
        :param v: Values. *Shape:* :math:`(N,S,H,V)`.
        :returns: Attention output and optional weights.
        """
        if q_layout.packed or k_layout.packed:
            raise NotSupportedError(
                "Gemma3n conformer SDPA does not support packed batches."
            )

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q_len = q.size(-2)
        k_len = k.size(-2)

        q = q * self.per_dim_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        rel_indices = self._get_relative_indices(k)
        rel_keys = self.rel_k_embed(rel_indices)
        rel_keys = rel_keys[-q_len:]

        rel_weights = torch.einsum("nhsk,stk->nhst", q, rel_keys)
        rel_weights = rel_weights * (self.head_dim**-0.5)

        chunked_mask = self._create_chunked_local_mask(q_len, k_len, q.device)
        chunked_mask = chunked_mask.unsqueeze(0).unsqueeze(0)

        if chunked_mask is not None:
            rel_weights = rel_weights.masked_fill(chunked_mask == 0, float("-inf"))

        attns, attn_weights = naive_scaled_dot_product_attention(
            q, k, v, rel_weights, needs_weights=needs_weights
        )

        attns = torch.tanh(attns / self.logit_cap) * self.logit_cap

        attns = attns.transpose(-2, -3)

        return attns, attn_weights if needs_weights else None

    if TYPE_CHECKING:
        __call__ = forward

    def _get_relative_indices(self, k: Tensor) -> Tensor:
        """Compute Shaw relative position indices."""
        indices = torch.arange(k.size(-2), device=k.device).unsqueeze(0)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices, -self.max_left_rel_pos, self.max_right_rel_pos
        )

        return rel_indices + self.max_left_rel_pos

    def _create_chunked_local_mask(
        self, q_len: int, k_len: int, device: Device
    ) -> Tensor:
        """Create chunked local attention mask with left/right context."""
        pos_i = torch.arange(q_len, device=device).unsqueeze(1)
        pos_j = torch.arange(k_len, device=device).unsqueeze(0)

        distance = pos_i - pos_j

        within_left = distance <= self.left_context
        within_right = distance >= -self.right_context

        mask = within_left & within_right

        return mask.to(dtype=torch.bool)


def init_shaw_embedding(embed: StandardEmbedding) -> None:
    """Initialize Shaw embedding with Xavier uniform."""
    nn.init.xavier_uniform_(embed.weight)

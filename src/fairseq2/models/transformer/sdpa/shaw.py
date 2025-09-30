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
from fairseq2.error import InternalError, NotSupportedError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.models.transformer.sdpa.naive import (
    naive_scaled_dot_product_attention,
)
from fairseq2.nn import BatchLayout, StandardEmbedding


@final
class ShawRelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1803.02155`."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        bias: AttentionBias,
        max_lhs_rel_pos: int,
        *,
        max_rhs_rel_pos: int | None = None,
        use_rel_pos_values: bool = False,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: The dimensionality of the model.
        :param: num_heads: The number of attention heads.
        :param: max_lhs_rel_pos: The left clipping value for relative positions.
        :param: max_rhs_rel_pos: The right clipping value for relative positions.
        :param: use_rel_pos_values: If ``True``, uses relative position values
            to compute attention.
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim

        self.num_heads = num_heads

        self.bias = bias

        if max_rhs_rel_pos is None:
            max_rhs_rel_pos = max_lhs_rel_pos

        self.max_lhs_rel_pos = max_lhs_rel_pos
        self.max_rhs_rel_pos = max_rhs_rel_pos

        num_pos = self.max_lhs_rel_pos + 1 + self.max_rhs_rel_pos

        head_dim = model_dim // num_heads

        self.rel_k_embed = StandardEmbedding(
            num_pos, head_dim, init_fn=init_shaw_embedding, device=device, dtype=dtype
        )

        if use_rel_pos_values:
            rel_v_embed = StandardEmbedding(
                num_pos,
                head_dim,
                init_fn=init_shaw_embedding,
                device=device,
                dtype=dtype,
            )
        else:
            rel_v_embed = None

        self.rel_v_embed: StandardEmbedding | None

        self.register_module("rel_v_embed", rel_v_embed)

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
        if q_layout.packed or k_layout.packed:
            raise NotSupportedError(
                f"`{ShawRelativePositionSDPA}` does not support packed batches."
            )

        # ([[N], H], S, S_kv)
        bias = maybe_get_attention_bias_tensor(
            self.bias, q, q_layout, k_layout, bias_cache
        )

        # (N, S, H, K) -> (N, H, S, K)
        q = q.transpose(-2, -3)

        # (N, S_kv, H, K) -> (N, H, S_kv, K)
        k = k.transpose(-2, -3)

        # (N, S_kv, H, V) -> (N, H, S_kv, V)
        v = v.transpose(-2, -3)

        q_len = q.size(-2)

        # (S_kv, S_kv)
        rel_indices = self._get_relative_indices(k)

        # (S_kv, S_kv, K_h)
        rel_keys = self.rel_k_embed(rel_indices)

        # (S_kv, S_kv, K_h) -> (S, S_kv, K_h)
        rel_keys = rel_keys[-q_len:]

        # (N, H, S, K_h) @ (S, S_kv, K_h) = (N, H, S, S_kv)
        rel_weights = torch.einsum("nhsk,stk->nhst", q, rel_keys)

        # We treat `rel_weights` as attention bias.
        rel_weights = rel_weights * (q.size(-1) ** -0.5)

        if bias is not None:
            rel_weights = rel_weights + bias

        attns, attn_weights = naive_scaled_dot_product_attention(
            q,
            k,
            v,
            rel_weights,
            needs_weights=needs_weights or self.rel_v_embed is not None,
        )

        if self.rel_v_embed is not None:
            if attn_weights is None:
                raise InternalError("`attn_weights` is `None`.")

            # (S_kv, S_kv, V_h)
            rel_pos_values = self.rel_v_embed(rel_indices)

            # (S_kv, S_kv, V_h) -> (S, S_kv, V_h)
            rel_pos_values = rel_pos_values[-q_len:]

            # (N, H, S, S_kv) @ (S, S_kv, V_h) = (N, H, S, V_h)
            rel_attns = torch.einsum("nhst,stv->nhsv", attn_weights, rel_pos_values)

            attns = attns + rel_attns

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, attn_weights if needs_weights else None

    if TYPE_CHECKING:
        __call__ = forward

    def _get_relative_indices(self, k: Tensor) -> Tensor:
        # (S, 1)
        indices = torch.arange(k.size(-2), device=k.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices, -self.max_lhs_rel_pos, self.max_rhs_rel_pos
        )

        return rel_indices + self.max_lhs_rel_pos

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias}, "
            f"max_lhs_rel_pos={self.max_lhs_rel_pos}, "
            f"max_rhs_rel_pos={self.max_rhs_rel_pos}"
        )


def init_shaw_embedding(embed: StandardEmbedding) -> None:
    """Initialize ``embed`` for use in :class:`ShawRelativePositionSDPA`."""
    nn.init.xavier_uniform_(embed.weight)

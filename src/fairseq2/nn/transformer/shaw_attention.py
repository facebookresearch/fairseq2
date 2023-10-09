# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import dropout, softmax

from fairseq2.nn.embedding import Embedding, StandardEmbedding
from fairseq2.nn.transformer.attention import SDPA
from fairseq2.typing import DataType, Device, finaloverride


@final
class ShawRelativePositionSDPA(SDPA):
    """Computes relative position scaled dot-product attention
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1803.02155`."""

    model_dim: int
    num_heads: int
    max_left_rel_pos: int
    max_right_rel_pos: Optional[int]
    rel_k_embed: Embedding
    rel_v_embed: Optional[Embedding]

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_pos: int,
        *,
        max_right_rel_pos: Optional[int] = None,
        use_rel_pos_values: bool = False,
        attn_dropout_p: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: max_left_rel_pos:
            The left clipping value for relative positions.
        :param: max_right_rel_pos:
            The right clipping value for relative positions.
        :param: use_rel_pos_values:
            If True, also uses relative position values to compute relative attention.
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__(attn_dropout_p=attn_dropout_p)

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        head_dim = model_dim // num_heads

        self.max_left_rel_pos = max_left_rel_pos
        self.max_right_rel_pos = (
            max_right_rel_pos if max_right_rel_pos is not None else max_left_rel_pos
        )
        num_pos = self.max_left_rel_pos + 1 + self.max_right_rel_pos

        self.rel_k_embed = StandardEmbedding(
            num_pos, head_dim, device=device, dtype=dtype
        )

        if use_rel_pos_values:
            self.rel_v_embed = StandardEmbedding(
                num_pos, head_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("rel_v_embed", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""

        assert isinstance(self.rel_k_embed.weight, Tensor)
        nn.init.xavier_uniform_(self.rel_k_embed.weight)

        if self.rel_v_embed is not None:
            assert isinstance(self.rel_v_embed.weight, Tensor)
            nn.init.xavier_uniform_(self.rel_v_embed.weight)

    def rel_pos_indices(self, seq_len: int, device: Device) -> Tensor:
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        rel_dist = pos - pos.transpose(0, 1)
        rel_dist = torch.clamp(rel_dist, -self.max_left_rel_pos, self.max_right_rel_pos)
        return rel_dist + self.max_left_rel_pos

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        mask: Optional[Tensor] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "`ShawRelativePositionSDPA` can only be used as part of a multi-head attention layer and expects its input tensors to be 4 dimensional."
            )

        # (N, H, S, head_dim) @ (N, H, head_dim, S_kv) = (N, H, S, S_kv)
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2))

        query_len, kv_len = queries.size(2), keys.size(2)

        # (S_kv, S_kv)
        rel_pos_indices = self.rel_pos_indices(kv_len, queries.device)

        # (S, S_kv, head_dim)
        rel_pos_keys = self.rel_k_embed(rel_pos_indices)[-query_len:]

        # (N, H, S, head_dim) @ (S, S_kv, head_dim) = (N, H, S, S_kv)
        rel_attn_weights = torch.einsum("nhsm,stm->nhst", queries, rel_pos_keys)

        attn_weights += rel_attn_weights

        attn_weights = attn_weights * (queries.size(-1) ** -0.5)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = attn_weights.type_as(queries)

        if self.training and self.attn_dropout_p > 0.0:
            attn_weights = dropout(attn_weights, self.attn_dropout_p)

        # (N, H, S, S_kv) @ (N, H, S_kv, head_dim) = (N, H, S, head_dim)
        attn = torch.matmul(attn_weights, values)

        if self.rel_v_embed is not None:
            # (S, S_kv, head_dim)
            rel_pos_values = self.rel_v_embed(rel_pos_indices)[-query_len:]

            # (N, H, S, S_kv) @ (S, S_kv, head_dim) = (N, H, S, head_dim)
            rel_attn = torch.einsum("nhst,stm->nhsm", attn_weights, rel_pos_values)

            attn += rel_attn

        return attn, attn_weights if needs_weights else None

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return (
            f"{s}, "
            f"model_dim={self.model_dim}, "
            f"num_heads={self.num_heads}, "
            f"max_left_rel_pos={self.max_left_rel_pos}, "
            f"max_right_rel_pos={self.max_right_rel_pos}"
        )

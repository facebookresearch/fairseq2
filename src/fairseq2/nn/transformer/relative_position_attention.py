# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding
from torch.nn.functional import dropout, softmax

from fairseq2.nn.transformer.attention import SDPA
from fairseq2.typing import DataType, Device, finaloverride


@final
class ShawRelativePositionSDPA(SDPA):
    """Computes relative position scaled dot-product attention
    as described in :cite:t:`https://arxiv.org/pdf/1803.02155v2.pdf`."""

    model_dim: int
    num_heads: int
    max_left_rel_position: int
    max_right_rel_position: Optional[int]
    rel_k_embedding: Embedding
    rel_v_embedding: Optional[Embedding]
    device: Optional[Device]

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_position: int,
        *,
        max_right_rel_position: Optional[int] = None,
        use_rel_position_values: bool = False,
        attn_dropout_p: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: max_left_rel_position:
            The left clipping value for relative positions.
        :param: max_right_rel_position:
            The right clipping value for relative positions.
        :param: use_rel_position_values:
            Whether to use relative position values to compute relative attention.
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

        self.max_left_rel_position = max_left_rel_position
        self.max_right_rel_position = (
            max_right_rel_position
            if max_right_rel_position is not None
            else max_left_rel_position
        )
        num_positions = self.max_left_rel_position + 1 + self.max_right_rel_position

        self.rel_k_embedding = Embedding(
            num_positions, head_dim, device=device, dtype=dtype
        )

        if use_rel_position_values:
            self.rel_v_embedding = Embedding(
                num_positions, head_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("rel_v_embedding", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.xavier_uniform_(self.rel_k_embedding.weight)
        if self.rel_v_embedding is not None:
            nn.init.xavier_uniform_(self.rel_v_embedding.weight)

    def rel_position_indices(self, seq_len: int) -> Tensor:
        positions = torch.arange(seq_len).unsqueeze(0)
        rel_dist = positions - positions.t()
        rel_dist = torch.clamp(
            rel_dist, -self.max_left_rel_position, self.max_right_rel_position
        )
        return rel_dist + self.max_left_rel_position

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

        query_length, kv_length = queries.shape[2], keys.shape[2]

        # (S_kv, S_kv)
        rel_position_indices = self.rel_position_indices(kv_length)

        rel_position_indices = rel_position_indices.to(device=queries.device)

        # (S, S_kv, head_dim)
        rel_position_keys = self.rel_k_embedding(rel_position_indices)[-query_length:]

        # (N, H, S, head_dim) @ (S, S_kv, head_dim) = (N, H, S, S_kv)
        rel_attn_weights = torch.einsum("nhsm,stm->nhst", queries, rel_position_keys)

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

        if self.rel_v_embedding is not None:
            # (S, S_kv, head_dim)
            rel_position_values = self.rel_v_embedding(rel_position_indices)[
                -query_length:
            ]

            # (N, H, S, S_kv) @ (S, S_kv, head_dim) = (N, H, S, head_dim)
            rel_attn = torch.einsum("nhst,stm->nhsm", attn_weights, rel_position_values)

            attn += rel_attn

        return attn, attn_weights if needs_weights else None

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, model_dim={self.model_dim}, num_heads={self.num_heads}"

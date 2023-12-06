# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor

from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.attention import SDPA, create_default_sdpa
from fairseq2.nn.transformer.attention_mask import AttentionMask, CustomAttentionMask
from fairseq2.typing import DataType, Device, finaloverride


@final
class ShawRelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1803.02155`."""

    model_dim: int
    num_heads: int
    max_left_rel_pos: int
    max_right_rel_pos: int
    rel_k_embed: StandardEmbedding
    rel_v_embed: Optional[StandardEmbedding]
    inner_sdpa: SDPA

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_pos: int,
        *,
        max_right_rel_pos: Optional[int] = None,
        use_rel_pos_values: bool = False,
        inner_sdpa: Optional[SDPA] = None,
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
            If ``True``, uses relative position values to compute attention.
        :param inner_sdpa:
            The actual :class:`SDPA` module to compute head attentions.
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        head_dim = model_dim // num_heads

        self.max_left_rel_pos = max_left_rel_pos

        self.max_right_rel_pos = (
            max_left_rel_pos if max_right_rel_pos is None else max_right_rel_pos
        )

        num_pos = self.max_left_rel_pos + 1 + self.max_right_rel_pos

        self.rel_k_embed = StandardEmbedding(
            num_pos, head_dim, init_fn=init_shaw_embedding, device=device, dtype=dtype
        )

        if use_rel_pos_values:
            self.rel_v_embed = StandardEmbedding(
                num_pos,
                head_dim,
                init_fn=init_shaw_embedding,
                device=device,
                dtype=dtype,
            )
        else:
            self.register_module("rel_v_embed", None)

        if inner_sdpa is not None:
            self.inner_sdpa = inner_sdpa
        else:
            self.inner_sdpa = create_default_sdpa()

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q_len = seqs.size(2)

        # (N, H, S, K_h) @ (N, H, K_h, S_kv) = (N, H, S, S_kv)
        attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

        # (S_kv, S_kv)
        rel_indices = self._get_relative_indices(keys)

        # (S_kv, S_kv, K_h)
        rel_keys = self.rel_k_embed(rel_indices)

        # (S_kv, S_kv, K_h) -> (S, S_kv, K_h)
        rel_keys = rel_keys[-q_len:]

        # (N, H, S, K_h) @ (S, S_kv, K_h) = (N, H, S, S_kv)
        rel_attn_weights = torch.einsum("nhsk,stk->nhst", seqs, rel_keys)

        # We treat `rel_attn_weights` as an attention mask to take advantage of
        # efficient SDPA implementations.
        rel_attn_weights = rel_attn_weights * (seqs.size(-1) ** -0.5)

        if attn_mask is None:
            mask = rel_attn_weights
        else:
            mask = rel_attn_weights + attn_mask.materialize()

        attn_mask = CustomAttentionMask(mask)

        attn, attn_weights = self.inner_sdpa(  # type: ignore[no-any-return]
            seqs,
            keys,
            key_padding_mask,
            values,
            attn_mask=attn_mask,
            needs_weights=needs_weights or self.rel_v_embed is not None,
        )

        if self.rel_v_embed is not None:
            assert attn_weights is not None

            # (S_kv, S_kv, V_h)
            rel_pos_values = self.rel_v_embed(rel_indices)

            # (S_kv, S_kv, V_h) -> (S, S_kv, V_h)
            rel_pos_values = rel_pos_values[-q_len:]

            # (N, H, S, S_kv) @ (S, S_kv, V_h) = (N, H, S, V_h)
            rel_attn = torch.einsum("nhst,stv->nhsv", attn_weights, rel_pos_values)

            attn = attn + rel_attn

        return attn, attn_weights if needs_weights else None

    def _get_relative_indices(self, keys: Tensor) -> Tensor:
        # (S, 1)
        indices = torch.arange(keys.size(2), device=keys.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices, -self.max_left_rel_pos, self.max_right_rel_pos
        )

        return rel_indices + self.max_left_rel_pos

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_heads={self.num_heads}, "
            f"max_left_rel_pos={self.max_left_rel_pos}, "
            f"max_right_rel_pos={self.max_right_rel_pos}"
        )


def init_shaw_embedding(embed: StandardEmbedding) -> None:
    """Initialize ``embed`` for use in :class:`ShawRelativePositionSDPA`."""
    nn.init.xavier_uniform_(embed.weight)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import pad
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.models.transformer.sdpa.naive import (
    naive_scaled_dot_product_attention,
)
from fairseq2.nn import BatchLayout, Linear


@final
class RelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1901.02860`."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        pos_encoding: RelativePositionalEncoding,
        bias: AttentionBias,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim

        self.num_heads = num_heads

        self.bias = bias

        self.pos_encoding = pos_encoding

        head_dim = model_dim // num_heads

        self.u_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )
        self.v_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )

        self.r_proj = Linear(
            model_dim, model_dim, bias=False, device=device, dtype=dtype
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.u_bias)
        nn.init.xavier_normal_(self.v_bias)

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
                f"`{RelativePositionSDPA}` does not support packed batches."
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

        # (H, K_h) -> (H, 1, K_h)
        u_bias = self.u_bias.unsqueeze(1)
        v_bias = self.v_bias.unsqueeze(1)

        # (N, H, S, K_h) + (H, 1, K_h) -> (N, H, S, K_h)
        q_with_u_bias = q + u_bias
        q_with_v_bias = q + v_bias

        # (N, H, 2 x S - 1, K_h)
        r = self._compute_r(k, batch_size=q.size(0))

        # (N, H, S, K_h) @ (N, H, K_h, 2 x S - 1) = (N, H, S, 2 x S - 1)
        bd = torch.matmul(q_with_v_bias, r.transpose(-1, -2))

        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        bd = self._shift_bd(bd)

        # We treat `bd` as attention bias.
        bd = bd * (q.size(-1) ** -0.5)

        if bias is not None:
            bd = bd + bias

        attns, attn_weights = naive_scaled_dot_product_attention(
            q_with_u_bias, k, v, bias=bd, needs_weights=needs_weights
        )

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, attn_weights

    def _compute_r(self, k: Tensor, batch_size: int) -> Tensor:
        # (2 x S - 1, K)
        r = self.pos_encoding(k)

        # (2 x S - 1, K) -> (2 x S - 1, K)
        r = self.r_proj(r)

        # (2 x S - 1, K) -> (1, 2 x S - 1, H, K_h)
        r = r.view(1, -1, self.num_heads, k.size(-1))

        # (1, 2 x S - 1, H, K_h) -> (N, H, 2 x S - 1, K_h)
        r = r.transpose(1, 2).expand(batch_size, -1, -1, -1)

        return r

    def _shift_bd(self, bd: Tensor) -> Tensor:
        # (N, H, S, 2 x S - 1) -> (N, H, S, 2 x S)
        x = pad(bd, (1, 0))

        # (N, H, S, 2 x S) -> (N, H, 2 x S, S)
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(2))

        # Discard the first set of positive positions.
        # (N, H, 2 x S, S) -> (N, H, 2 x S - 1, S)
        x = x[:, :, 1:, :]

        # This op effectively shifts each row by an extra step.
        # (N, H, 2 x S - 1, S) -> (N, H, S, 2 x S - 1)
        x = x.view_as(bd)

        # Discard positions used for shift.
        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        x = x[..., : bd.size(2)]

        return x

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, num_heads={self.num_heads}, bias={self.bias}"
        )


@final
class RelativePositionalEncoding(Module):
    """Produces relative positional encodings as described in Appendix B of
    :cite:t:`dai2019transformerxl`."""

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim:
            The dimensionality of positional encodings.
        :param max_seq_len:
            The maximum sequence length.
        """
        super().__init__()

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        self.encoding_dim = encoding_dim

        self.max_seq_len = max_seq_len

        freqs = torch.empty(
            ((max_seq_len * 2) - 1, encoding_dim), device=device, dtype=torch.float32
        )

        self.freqs: Tensor

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        device, dtype = self.freqs.device, self.freqs.dtype

        positive_half = self.freqs[: self.max_seq_len]
        negative_half = self.freqs[self.max_seq_len :]

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (E / 2)
        indices = torch.arange(0, self.encoding_dim, step=2, device=device, dtype=dtype)

        freqs = torch.exp(indices * -math.log(10000.0) / self.encoding_dim)

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        flipped_freqs = freqs.flip([0])

        # A mirrored matrix of sinusoidal positive and negative positional
        # encodings to use in shift trick.
        #
        # [max, ...,  3,  2,  1,  0, -1, -2, -3, ..., min]
        torch.sin(flipped_freqs, out=positive_half[:, 0::2])
        torch.cos(flipped_freqs, out=positive_half[:, 1::2])

        torch.sin(-1 * freqs[1:], out=negative_half[:, 0::2])
        torch.cos(-1 * freqs[1:], out=negative_half[:, 1::2])

    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to return positional encodings. *Shape:*
            :math:`(*,S,E)`, where :math:`*` is any number of batch dimensions
            including none, :math:`S` is the sequence length, and :math:`E` is
            the dimensionality of the positional encodings.

        :returns:
            The positional encodings to use in shift trick in
            :class:`RelativePositionSDPA`. *Shape:* :math:`(2 x S - 1, E)`,
            where :math:`S` is the sequence length and :math:`E` is the
            dimensionality of the positional encodings.
        """
        seq_len = seqs.size(-2)

        max_seq_len = self.max_seq_len

        if seq_len > max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({max_seq_len}), but at least one sequence is of length {seq_len} instead."
            )

        fp32_freqs = self.freqs[max_seq_len - seq_len : max_seq_len + seq_len - 1]

        return fp32_freqs.type_as(seqs)

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"

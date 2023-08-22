# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import dropout, pad, softmax

from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer.attention import SDPA
from fairseq2.typing import DataType, Device, finaloverride


@final
class RelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1901.02860`."""

    model_dim: int
    num_heads: int
    pos_encoding: "RelativePositionalEncoding"
    u_bias: Parameter
    v_bias: Parameter
    r_proj: Linear

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        pos_encoding: "RelativePositionalEncoding",
        attn_dropout_p: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: pos_encoding:
            The relative positional encoding table.
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__(attn_dropout_p)

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be divisible by `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        if pos_encoding.encoding_dim != model_dim:
            raise ValueError(
                f"`encoding_dim` of `pos_encoding` must be equal `model_dim` ({model_dim}), but is {pos_encoding.encoding_dim} instead."
            )

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
        """Reset the parameters and buffers of the module."""
        nn.init.xavier_normal_(self.u_bias)
        nn.init.xavier_normal_(self.v_bias)

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = queries
        k = keys

        # (H, K_h) -> (H, 1, K_h)
        u_bias = self.u_bias.unsqueeze(1)
        v_bias = self.v_bias.unsqueeze(1)

        # (N x H, S, K_h) -> (N, H, S, K_h)
        q = q.unflatten(0, (-1, self.num_heads))

        # (N, H, S, K_h) + (H, 1, K_h) -> (N, H, S, K_h)
        q_with_u_bias = q + u_bias
        q_with_v_bias = q + v_bias

        # (N, H, S, K_h) -> (N x H, S, K_h)
        q_with_u_bias = q_with_u_bias.flatten(0, 1)
        q_with_v_bias = q_with_v_bias.flatten(0, 1)

        # (N x H, 2 x S - 1, K_h)
        r = self._compute_r(k, batch_size=q.size(0))

        # (N x H, S, K_h) @ (N x H, K_h, S) = (N x H, S, S)
        ac = torch.bmm(q_with_u_bias, k.transpose(1, 2))

        # (N x H, S, K_h) @ (N x H, K_h, 2 x S - 1) = (N x H, S, 2 x S - 1)
        bd = torch.bmm(q_with_v_bias, r.transpose(1, 2))

        # (N x H, S, 2 x S -1) -> (N x H, S, S)
        bd = self._shift_bd(bd)

        attn_weights = (ac + bd) * (q.size(-1) ** -0.5)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = attn_weights.type_as(queries)

        if self.training and self.attn_dropout_p > 0.0:
            attn_weights = dropout(attn_weights, self.attn_dropout_p)

        # (N x H, S, S) @ (N x H, S, V_h) = (N x H, S, V_h)
        attn = torch.bmm(attn_weights, values)

        return attn, attn_weights if needs_weights else None

    def _compute_r(self, k: Tensor, batch_size: int) -> Tensor:
        # (S, K) -> (2 x S - 1, K)
        r = self.pos_encoding(k)

        # (2 x S - 1, K) -> (2 x S - 1, K)
        r = self.r_proj(r)

        # (2 x S - 1, K) -> (1, 2 x S - 1, H, K_h)
        r = r.view(1, -1, self.num_heads, k.size(2))

        # (1, 2 x S - 1, H, K_h) -> (N, H, 2 x S - 1, K_h)
        r = r.transpose(1, 2).expand(batch_size, -1, -1, -1)

        # (N, H, 2 x S - 1, K_h) -> (N x H, 2 x S - 1, K_h)
        r = r.flatten(0, 1)

        return r  # type: ignore[no-any-return]

    def _shift_bd(self, bd: Tensor) -> Tensor:
        # (N x H, S, 2 x S - 1) -> (N x H, S, 2 x S)
        x = pad(bd, (1, 0))

        # (N x H, S, 2 x S) -> (N x H, 2 x S, S)
        x = x.view(x.size(0), x.size(2), x.size(1))

        # Discard the first set of positive positions.
        # (N x H, 2 x S, S) -> (N x H, 2 x S - 1, S)
        x = x[:, 1:, :]

        # This op effectively shifts each row by an extra step.
        # (N x H, S, 2 x S - 1)
        x = x.view_as(bd)

        # Discard positions used for shift.
        # (N x H, S, 2 x S - 1) -> (N x H, S, S)
        x = x[..., : bd.size(1)]

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", model_dim={self.model_dim}, num_heads={self.num_heads}"


class RelativePositionalEncoding(Module):
    """Produces relative positional encodings as described in Appendix B of
    :cite:t:`dai2019transformerxl`."""

    encoding_dim: int
    max_seq_len: int
    weight: Tensor

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param encoding_dim:
            The dimensionality of positional encodings.
        :param max_seq_len:
            The expected maximum sequence length.
        """
        super().__init__()

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

        weight = torch.empty(
            ((max_seq_len * 2) - 1, encoding_dim), device=device, dtype=dtype
        )

        self.register_buffer("weight", weight, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        dtype = torch.float32

        weight = self.weight.to(dtype)

        positive_w = weight[: self.max_seq_len]
        negative_w = weight[self.max_seq_len :]

        device = weight.device

        # (E / 2)
        indices = torch.arange(0, self.encoding_dim, step=2, device=device, dtype=dtype)

        # (1, E / 2)
        indices = indices.unsqueeze(0)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (S, 1)
        steps = steps.unsqueeze(1)

        factors = torch.exp(indices * -math.log(10000) / self.encoding_dim)

        # (S, 1) x (1, E / 2) -> (S, E / 2)
        factors = torch.matmul(steps, factors)

        flipped_factors = factors.flip([0])

        # A mirrored matrix of sinusoidal positive and negative positional
        # encodings to use in shift trick.
        #
        # [max, ...,  3,  2,  1,  0, -1, -2, -3, ..., min]
        torch.sin(flipped_factors, out=positive_w[:, 0::2])
        torch.cos(flipped_factors, out=positive_w[:, 1::2])

        torch.sin(-1 * factors[1:], out=negative_w[:, 0::2])
        torch.cos(-1 * factors[1:], out=negative_w[:, 1::2])

        self.weight.copy_(weight)

    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to return positional encodings. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            The positional encodings to use in shift trick in
            :class:`RelativePositionSDPA`. *Shape:* :math:`(2 x S - 1, E)`,
            where :math:`S` is the sequence length and :math:`E` is the
            dimensionality of the positional encodings.
        """
        seq_len = seqs.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
            )

        return self.weight[self.max_seq_len - seq_len : self.max_seq_len + seq_len - 1]

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"

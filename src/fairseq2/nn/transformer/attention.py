# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import final as finaloverride
from packaging import version
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.position_encoder import PositionEncoder

log = logging.getLogger(__name__)

is_pt2_or_greater = version.parse(torch.__version__) >= version.parse("2.0.0")

has_warned_sdpa = False


def torch_scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute scaled dot-product attention using PyTorch SDPA 2.0 as described
    `here <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
    in the PyTorch documentation.
    """
    if not is_pt2_or_greater:
        raise ValueError(
            "`torch_scaled_dot_product_attention` requires PyTorch 2.0.0 or greater."
        )

    if needs_weights:
        raise ValueError(
            "`torch_scaled_dot_product_attention` does not support the `needs_weights` parameter."
        )

    if not training:
        dropout_p = 0.0

    # Check if the mask is tagged as causal.
    is_causal: bool = getattr(mask, "is_causal", False)

    attn = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
        queries,
        keys,
        values,
        attn_mask=None if is_causal else mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

    return attn, None


def naive_scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    needs_weights: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute scaled dot-product attention using a non-fused Python-based
    implementation.

    This function is used internally by :func:`default_scaled_dot_product_attention`
    as a fall-back if no other efficient implementation is available.

    It can also be used explicitly for troubleshooting purposes since it allows
    step-through debugging.
    """
    queries = queries * (queries.size(-1) ** -0.5)

    if mask is None:
        # (N, S, K) @ (N, K, S_kv) = (N, S, S_kv)
        attn_weights = torch.bmm(queries, keys.transpose(1, 2))
    else:
        # (N, S, S_kv) + ((N, S, K) @ (N, K, S_kv)) = (N, S, S_kv)
        attn_weights = torch.baddbmm(mask, queries, keys.transpose(1, 2))

    attn_weights = F.softmax(attn_weights, dim=-1)

    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, dropout_p, training)

    # (N, S, S_kv) @ (N, S_kv, V) = (N, S, V)
    attn = torch.bmm(attn_weights, values)

    return attn, attn_weights if needs_weights else None


class SDPA(Module, ABC):
    """Computes attention."""

    model_dim: int
    num_heads: int

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads
        """
        self.model_dim = model_dim
        self.num_heads = num_heads

    @abstractmethod
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        needs_weights: bool = False,
        training: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param queries:
            The queries. *Shape:* :math:`(N,S,K)`, where :math:`N` is the batch
            size, :math:`S` is the sequence length, and :math:`K` is the key
            size.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(S,S_{kv})` or
            :math:`(N,S,S_{kv})`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`S_{kv}` is the
            key/value sequence length.
        :param dropout_p:
            The dropout probability on the attention weights.
        :param needs_weights:
            If ``True``, returns the attention weights as the second item of the
            returned tuple.
        :param training:
            If ``True``, applies dropout.

        :returns:
            - The attention values. *Shape:* :math:`(N,S,V)`, where
              :math:`N` is the batch size, :math:`S` is the sequence length, and
              :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`S` is the sequence length, and
              :math:`S_{kv}` is the key/value sequence length.
        """


class DefaultSDPA(SDPA):
    """Compute scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    This class automatically picks the most efficient SDPA implementation and
    uses it internally.
    """

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        needs_weights: bool = False,
        training: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        global has_warned_sdpa

        if not needs_weights and is_pt2_or_greater and queries.is_cuda:
            return torch_scaled_dot_product_attention(
                queries, keys, values, mask, dropout_p, needs_weights, training
            )
        else:
            if is_pt2_or_greater and queries.is_cuda and not has_warned_sdpa:
                log.warning(
                    "You are failing to leverage the more efficient `torch_scaled_dot_product_attention` that is based on PyTorch's native SDPA because of `needs_weights` set to `True`."
                )

                has_warned_sdpa = True

            return naive_scaled_dot_product_attention(
                queries, keys, values, mask, dropout_p, needs_weights, training
            )


class RelativePositionSDPA(SDPA):
    """Compute relative position scaled dot-product attention as described in
    :cite:t:`dai2019transformerxl`."""

    pos_encoder: PositionEncoder

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        pos_encoder: PositionEncoder,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: pos_encoder:
            The position encoder to generate the relative position tensor :math:`R`.
        """
        super().__init__(model_dim, num_heads)

        from fairseq2.nn.transformer.multihead_attention import QKVProjection

        pos_bias_shp = (1, num_heads, 1, model_dim // num_heads)
        self.u_bias = torch.nn.Parameter(
            torch.empty(pos_bias_shp, device=device, dtype=dtype)
        )
        self.v_bias = torch.nn.Parameter(
            torch.empty(pos_bias_shp, device=device, dtype=dtype)
        )

        self.pos_proj = QKVProjection(model_dim, device=device, dtype=dtype)
        self.R = self._generate_r_embed(pos_encoder, device, dtype)

        self.reset_parameters()

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        needs_weights: bool = False,
        training: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        queries = queries * (queries.size(-1) ** -0.5)

        # (N x H, S, K_h) -> (N, H, S, K_h)
        queries = queries.unflatten(0, (-1, self.num_heads))
        # (N, H, S, K_h) + (1, H, 1, K_h) -> (N x H, S, K_h)
        queries_u_bias = (queries + self.u_bias).flatten(0, 1)
        queries_v_bias = (queries + self.v_bias).flatten(0, 1)

        batch_heads, S_kv, _ = keys.size()
        batch_size = batch_heads // self.num_heads

        # (H, S_kv, K_h) -> (N x H, S_kv, K_h)
        R = self.R[:, :S_kv, :]
        R = R.repeat([batch_size, 1, 1])

        if mask is None:
            # (N x H, S, K_h) @ (N x H, K_h, S_kv) -> (N, S, S_kv)
            content_score = torch.bmm(queries_u_bias, keys.transpose(1, 2))
            pos_score = torch.bmm(queries_v_bias, R.transpose(1, 2))
        else:
            # (N x H, S, S_kv) + ((N x H, S, K_h) @ (N x H, K_h, S_kv)) -> (N x H, S, S_kv)
            content_score = torch.baddbmm(mask, queries_u_bias, keys.transpose(1, 2))
            pos_score = torch.baddbmm(mask, queries_v_bias, R.transpose(1, 2))

        pos_score = self._shift(pos_score)

        attn_weights = content_score + pos_score
        attn_weights = F.softmax(attn_weights, dim=-1)

        if training and dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, dropout_p, training)

        # (N, S, S_kv) @ (N, S_kv, V) = (N, S, V)
        attn = torch.bmm(attn_weights, values)

        return attn, attn_weights if needs_weights else None

    def _shift(self, pos_score: Tensor) -> Tensor:
        NH, S, S_kv = pos_score.size()
        zero_pad = pos_score.new_zeros(NH, S, 1)

        padded_pos_score = torch.cat([zero_pad, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(NH, S_kv + 1, S)
        pos_score = pos_score = padded_pos_score[:, 1:].view_as(pos_score)

        return pos_score

    def _generate_r_embed(
        self,
        pos_encoder: PositionEncoder,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        max_seq_len = pos_encoder.max_seq_len
        if max_seq_len is None:
            raise ValueError(
                "pos_encoder must have the max_seq_len attribute defined for RelativePositionSDPA"
            )

        R = torch.zeros([1, max_seq_len, self.model_dim], device=device, dtype=dtype)
        R = pos_encoder(R)

        # (N, S, model_dim) -> (N, S, H, K_h)
        R = R.unflatten(-1, (self.num_heads, -1))
        # (N, S, H, K_h) -> (N, H, S, K_h)
        R = R.transpose(1, 2)
        # (N, H, S, K_h) -> (N x H, S, K_h)
        R = R.flatten(0, 1)
        # (N, S_kv, model_dim) -> (N x H, S, K_h)

        return R

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.u_bias)
        torch.nn.init.xavier_normal_(self.v_bias)

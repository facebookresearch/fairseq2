# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import final

import torch
from torch import Tensor
from torch.nn import Dropout, Module, Parameter
from typing_extensions import override

from fairseq2.models.jepa.model import JepaModel
from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    ResidualConnect,
    TransformerEncoder,
    TransformerNormOrder,
    make_standard_layer_norm,
)
from fairseq2.nn.transformer.residual import StandardResidualConnect
from fairseq2.typing import DataType, Device


class CrossAttentionDecoder(Module):
    """Represents a simple transformer decoder with only cross attention and layernorm"""

    model_dim: int
    cross_attn: MultiheadAttention
    cross_attn_dropout: Dropout | None
    cross_attn_residual: ResidualConnect | None
    cross_attn_layer_norm: LayerNorm | None
    ffn: FeedForwardNetwork
    ffn_dropout: Dropout | None
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        cross_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: LayerNormFactory | None = None,
        cross_attn_residual: ResidualConnect | None = None,
        ffn_residual: ResidualConnect | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param cross_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        :param cross_attn_residual:
            The residual connection between the input and output of the
            encoder-decoder attention layer.
        :param ffn_residual:
            The residual connection between the input and output of the
            feed-forward network.
            attention layer.
        """
        model_dim = cross_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = make_standard_layer_norm

        cross_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        if norm_order != TransformerNormOrder.POST:
            self.cross_attn_layer_norm = cross_attn_layer_norm

        self.cross_attn = cross_attn

        if dropout_p > 0.0:
            self.cross_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("cross_attn_dropout", None)

        if cross_attn_residual is None:
            cross_attn_residual = StandardResidualConnect()

        self.cross_attn_residual = cross_attn_residual

        if norm_order == TransformerNormOrder.POST:
            self.cross_attn_layer_norm = cross_attn_layer_norm

        ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if ffn_residual is None:
            ffn_residual = StandardResidualConnect()

        self.ffn_residual = ffn_residual

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

    @override
    def forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        encoder_output: Tensor | None = None,
        encoder_padding_mask: PaddingMask | None = None,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, PaddingMask | None]:
        if encoder_output is None:
            raise ValueError(
                "`encoder_output` must not be `None` for encoder-decoder attention."
            )

        seqs = self._forward_cross_attn(
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

        seqs = self._forward_ffn(seqs)
        return seqs, padding_mask

    def _forward_cross_attn(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        encoder_output: Tensor | None,
        encoder_padding_mask: PaddingMask | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:

        residual = seqs

        assert self.cross_attn_residual is not None
        assert self.cross_attn_layer_norm is not None

        # Note that the cross-attention norm is applief on encoder output and not seqs
        if self.norm_order != TransformerNormOrder.POST:
            encoder_output = self.cross_attn_layer_norm(encoder_output)

        seqs = self.cross_attn(
            seqs,
            padding_mask,
            keys=encoder_output,
            key_padding_mask=encoder_padding_mask,
            values=encoder_output,
            state_bag=state_bag,
        )

        if self.cross_attn_dropout is not None:
            seqs = self.cross_attn_dropout(seqs)

        seqs = self.cross_attn_residual(seqs, residual)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.cross_attn_layer_norm(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        seqs = self.ffn_residual(seqs, residual)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, model_dim={self.model_dim}, norm_order={self.norm_order.name}"


@final
class AttentivePooler(Module):
    """
    An attentive pooler that gets output of a Jepa encoder and decode it into
    a logit of a given task.

    TODO:
    - Move this into fairseq2.nn to benefit other similiar tasks. Internally,
    this module is just a thin transformer encoder without self attention layer.
    Optionally, it can consist of some extra transformer encoders depending on the
    (finetuning) task
    """

    model_dim: int
    num_queries: int
    decoder: CrossAttentionDecoder
    encoder: TransformerEncoder | None
    init_fn: Callable[[Tensor], None] | None

    def __init__(
        self,
        decoder: CrossAttentionDecoder,
        encoder: TransformerEncoder | None,
        *,
        num_queries: int = 1,
        init_fn: Callable[[Tensor], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim

        self.decoder = decoder

        if encoder:
            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        self.num_queries = num_queries
        self.query_tokens = Parameter(
            torch.empty(1, num_queries, self.model_dim, device=device, dtype=dtype)
        )

        if init_fn:
            init_fn(self.pool_layer)

    def forward(
        self, seqs: Tensor, padding_mask: PaddingMask | None
    ) -> tuple[Tensor, PaddingMask | None]:
        if self.encoder:
            seqs, padding_mask = self.encoder(seqs, padding_mask)
        queries = self.query_tokens.repeat(len(seqs), 1, 1)
        seqs, padding_mask = self.decoder(queries, None, seqs, padding_mask)
        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, model_dim={self.model_dim}, pools={self.num_queries}"


@final
class JepaForClassification(Model):
    """
    Represents a pretrained Jepa model, with an attentive probing layer for
    classfication tasks. See
        * :cite:t:`https://doi.org/10.48550/arXiv.2301.08243`
        * :cite:t:`https://doi.org/10.48550/arXiv.2404.08471`
    """

    model_dim: int
    encoder: JepaModel
    pooler: AttentivePooler
    head: Projection

    def __init__(
        self,
        encoder: JepaModel,
        pooler: AttentivePooler,
        head: Projection,
    ) -> None:
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder = encoder
        self.pooler = pooler
        self.head = head

    def forward(self, batch: SequenceBatch) -> Tensor:
        encoder_output: SequenceBatch = self.encoder(batch)
        seqs, _ = self.pooler(encoder_output.seqs, encoder_output.padding_mask)
        seqs = seqs.squeeze(1)
        output: Tensor = self.head(seqs)
        return output

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, model_dim={self.model_dim}"

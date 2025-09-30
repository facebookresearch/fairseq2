# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Dropout, Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.models.transformer.ffn import FeedForwardNetwork
from fairseq2.models.transformer.multihead_attention import MultiheadAttention
from fairseq2.models.transformer.norm_order import TransformerNormOrder
from fairseq2.nn import (
    AdditiveResidualConnect,
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    ResidualConnect,
)


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs: The sequences to process. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        :param encoder_output: The encoder output to use in encoder-decoder
            attention. *Shape:* :math:`(N,S_{enc},M_{enc})`, where :math:`N` is
            the batch size, :math:`S_{enc}` is the encoder output sequence
            length, and :math:`M_{enc}` is the dimensionality of the encoder.
        :param state_bag: The state bag to use for incremental decoding.

        :returns: The decoder layer output. *Shape:* Same as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    def __init__(
        self,
        self_attn: MultiheadAttention,
        self_attn_layer_norm: LayerNorm,
        encoder_decoder_attn: MultiheadAttention,
        encoder_decoder_attn_layer_norm: LayerNorm,
        ffn: FeedForwardNetwork,
        ffn_layer_norm: LayerNorm,
        *,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        self_attn_residual: ResidualConnect | None = None,
        encoder_decoder_attn_residual: ResidualConnect | None = None,
        ffn_residual: ResidualConnect | None = None,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param self_attn: The self attention layer.
        :param encoder_decoder_attn: The encoder-decoder attention layer.
        :param ffn: The feed-forward network.
        :param norm_order: The Layer Normalization order.
        :param layer_norm_factory: The factory to construct the Layer
            Normalization modules.
        :param self_attn_residual: The residual connection between the input and
            output of the self attention layer.
        :param encoder_decoder_attn_residual: The residual connection between
            the input and output of the encoder-decoder attention layer.
        :param ffn_residual: The residual connection between the input and
            output of the feed-forward network.
        :param dropout_p: The dropout probability on outputs of the attention
            layers and the feed-forward network.
        """
        super().__init__()

        # Self Attention
        self.self_attn_layer_norm: LayerNorm

        if norm_order != TransformerNormOrder.POST:
            self.register_module("self_attn_layer_norm", self_attn_layer_norm)

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self_attn_dropout = Dropout(dropout_p)
        else:
            self_attn_dropout = None

        self.self_attn_dropout: Dropout | None

        self.register_module("self_attn_dropout", self_attn_dropout)

        if self_attn_residual is None:
            self_attn_residual = AdditiveResidualConnect()

        self.self_attn_residual = self_attn_residual

        if norm_order == TransformerNormOrder.POST:
            self.register_module("self_attn_layer_norm", self_attn_layer_norm)

        # Encoder-Decoder Attention
        self.encoder_decoder_attn_layer_norm: LayerNorm

        if norm_order != TransformerNormOrder.POST:
            self.register_module(
                "encoder_decoder_attn_layer_norm", encoder_decoder_attn_layer_norm
            )

        self.encoder_decoder_attn = encoder_decoder_attn

        if dropout_p > 0.0:
            encoder_decoder_attn_dropout = Dropout(dropout_p)
        else:
            encoder_decoder_attn_dropout = None

        self.encoder_decoder_attn_dropout: Dropout | None

        self.register_module(
            "encoder_decoder_attn_dropout", encoder_decoder_attn_dropout
        )

        if encoder_decoder_attn_residual is None:
            encoder_decoder_attn_residual = AdditiveResidualConnect()

        self.encoder_decoder_attn_residual = encoder_decoder_attn_residual

        if norm_order == TransformerNormOrder.POST:
            self.register_module(
                "encoder_decoder_attn_layer_norm", encoder_decoder_attn_layer_norm
            )

        # Feed-Forward Network
        self.ffn_layer_norm: LayerNorm

        if norm_order != TransformerNormOrder.POST:
            self.register_module("ffn_layer_norm", ffn_layer_norm)

        self.ffn = ffn

        if dropout_p > 0.0:
            ffn_dropout = Dropout(dropout_p)
        else:
            ffn_dropout = None

        self.ffn_dropout: Dropout | None

        self.register_module("ffn_dropout", ffn_dropout)

        if ffn_residual is None:
            ffn_residual = AdditiveResidualConnect()

        self.ffn_residual = ffn_residual

        if norm_order == TransformerNormOrder.POST:
            self.register_module("ffn_layer_norm", ffn_layer_norm)

        self.norm_order = norm_order

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        seqs = self._forward_self_attn(seqs, seqs_layout, attn_bias_cache, state_bag)

        seqs = self._forward_encoder_decoder_attn(
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            attn_bias_cache,
            state_bag,
        )

        seqs = self._forward_ffn(seqs)

        return seqs

    def _forward_self_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            keys_layout=seqs_layout,
            values=seqs,
            bias_cache=attn_bias_cache,
            state_bag=state_bag,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = self.self_attn_residual(seqs, residual)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_encoder_decoder_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.encoder_decoder_attn_layer_norm(seqs)

        seqs = self.encoder_decoder_attn(
            seqs,
            seqs_layout,
            keys=encoder_output,
            keys_layout=encoder_output_layout,
            values=encoder_output,
            bias_cache=attn_bias_cache,
            state_bag=state_bag,
        )

        if self.encoder_decoder_attn_dropout is not None:
            seqs = self.encoder_decoder_attn_dropout(seqs)

        seqs = self.encoder_decoder_attn_residual(seqs, residual)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.encoder_decoder_attn_layer_norm(seqs)

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

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"norm_order={self.norm_order.name}"

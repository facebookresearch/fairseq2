# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from torch import Tensor
from torch.nn import Dropout, Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    ResidualConnect,
    StandardResidualConnect,
)

# isort: split

from fairseq2.models.transformer._attention_bias import AttentionBiasCache
from fairseq2.models.transformer._ffn import FeedForwardNetwork
from fairseq2.models.transformer._multihead_attention import MultiheadAttention
from fairseq2.models.transformer._norm_order import TransformerNormOrder
from fairseq2.models.transformer._normalization import (
    LayerNormFactory,
    create_standard_layer_norm,
)


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim: The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

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

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_dropout: Dropout | None
    self_attn_residual: ResidualConnect
    self_attn_layer_norm: LayerNorm
    encoder_decoder_attn: MultiheadAttention
    encoder_decoder_attn_dropout: Dropout | None
    encoder_decoder_attn_residual: ResidualConnect
    encoder_decoder_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Dropout | None
    ffn_residual: ResidualConnect
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        encoder_decoder_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: LayerNormFactory | None = None,
        self_attn_residual: ResidualConnect | None = None,
        encoder_decoder_attn_residual: ResidualConnect | None = None,
        ffn_residual: ResidualConnect | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param self_attn: The self attention layer.
        :param encoder_decoder_attn: The encoder-decoder attention layer.
        :param ffn: The feed-forward network.
        :param dropout_p: The dropout probability on outputs of the attention
            layers and the feed-forward network.
        :param norm_order: The Layer Normalization order.
        :param layer_norm_factory: The factory to construct the Layer
            Normalization modules.
        :param self_attn_residual: The residual connection between the input and
            output of the self attention layer.
        :param encoder_decoder_attn_residual: The residual connection between
            the input and output of the encoder-decoder attention layer.
        :param ffn_residual: The residual connection between the input and
            output of the feed-forward network.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        # Self Attention
        self_attn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if self_attn_residual is None:
            self_attn_residual = StandardResidualConnect()

        self.self_attn_residual = self_attn_residual

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        # Encoder-Decoder Attention
        encoder_decoder_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        if norm_order != TransformerNormOrder.POST:
            self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

        self.encoder_decoder_attn = encoder_decoder_attn

        if dropout_p > 0.0:
            self.encoder_decoder_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("encoder_decoder_attn_dropout", None)

        if encoder_decoder_attn_residual is None:
            encoder_decoder_attn_residual = StandardResidualConnect()

        self.encoder_decoder_attn_residual = encoder_decoder_attn_residual

        if norm_order == TransformerNormOrder.POST:
            self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

        # Feed-Forward Network
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

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order.name}"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.models.s2t_transformer.config import S2TTransformerConfig
from fairseq2.models.s2t_transformer.feature_extractor import Conv1dFbankSubsampler
from fairseq2.models.s2t_transformer.frontend import S2TTransformerFrontend
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEmbeddingFrontend,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerFrontend,
    TransformerModel,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    StandardLayerNorm,
    init_scaled_embedding,
)


def create_s2t_transformer_model(config: S2TTransformerConfig) -> TransformerModel:
    return S2TTransformerFactory(config).create_model()


class S2TTransformerFactory:
    def __init__(self, config: S2TTransformerConfig) -> None:
        self._config = config

    def create_model(self) -> TransformerModel:
        config = self._config

        encoder_frontend = self.create_encoder_frontend()

        encoder = self.create_encoder()

        decoder_frontend = self.create_decoder_frontend()

        decoder = self.create_decoder()

        final_proj = self.create_final_projection()

        return TransformerModel(
            config.model_dim,
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_source_seq_len,
            config.max_target_seq_len,
        )

    def create_encoder_frontend(self) -> TransformerFrontend:
        config = self._config

        feat_extractor = Conv1dFbankSubsampler(
            num_channels=config.num_fbank_channels,
            inner_dim=1024,
            feature_dim=config.model_dim,
            kernel_sizes=[5, 5],
        )

        pos_encoder = self.create_source_position_encoder()

        return S2TTransformerFrontend(
            config.model_dim,
            feat_extractor,
            pos_encoder,
            proj=None,
            dropout_p=config.dropout_p,
        )

    def create_source_position_encoder(self) -> PositionEncoder:
        config = self._config

        return SinusoidalPositionEncoder(config.model_dim, config.max_source_seq_len)

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_encoder_layer()

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerEncoder(layers, layer_norm)

    def create_encoder_layer(self) -> TransformerEncoderLayer:
        self_attn = self.create_encoder_self_attention()

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn()

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerEncoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
        )

    def create_encoder_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = IdentityBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_encoder_attn_heads, sdpa
        )

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        return StandardFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=True,
            inner_dropout_p=config.dropout_p,
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        config = self._config

        embed = self.create_target_embedding()

        pos_encoder = self.create_target_position_encoder()

        return TransformerEmbeddingFrontend(
            config.model_dim, embed, pos_encoder, dropout_p=config.dropout_p
        )

    def create_target_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(
            config.target_vocab_size,
            config.model_dim,
            config.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def create_target_position_encoder(self) -> PositionEncoder:
        config = self._config

        return SinusoidalPositionEncoder(
            config.model_dim, config.max_target_seq_len, _legacy_pad_idx=1
        )

    def create_decoder(self) -> TransformerDecoder:
        config = self._config

        layers = []

        for _ in range(config.num_decoder_layers):
            layer = self.create_decoder_layer()

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerDecoder(layers, layer_norm)

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        self_attn = self.create_decoder_self_attention()

        self_attn_layer_norm = self.create_layer_norm()

        encoder_decoder_attn = self.create_encoder_decoder_attention()

        encoder_decoder_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn()

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            encoder_decoder_attn,
            encoder_decoder_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
        )

    def create_decoder_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_decoder_attn_heads, sdpa
        )

    def create_encoder_decoder_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = IdentityBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_decoder_attn_heads, sdpa
        )

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        return StandardLayerNorm(config.model_dim, bias=True)

    def create_final_projection(self) -> Projection:
        config = self._config

        return Linear(
            config.model_dim,
            config.target_vocab_size,
            bias=False,
            init_fn=_init_final_projection,
        )


def _init_final_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, std=proj.input_dim**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)

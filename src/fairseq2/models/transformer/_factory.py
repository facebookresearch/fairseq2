# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.models.transformer._config import TransformerConfig
from fairseq2.models.transformer._frontend import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.transformer._model import TransformerModel
from fairseq2.nn import (
    Embedding,
    Linear,
    PositionEncoder,
    Projection,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    TiedProjection,
    init_scaled_embedding,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_default_sdpa,
)


def create_transformer_model(config: TransformerConfig) -> TransformerModel:
    return TransformerFactory(config).create_model()


class TransformerFactory:
    _config: TransformerConfig

    def __init__(self, config: TransformerConfig) -> None:
        self._config = config

    def create_model(self) -> TransformerModel:
        config = self._config

        embed = self.create_embedding()

        frontend = self.create_frontend(embed)

        encoder = self.create_encoder()

        decoder = self.create_decoder()

        final_proj = self.create_final_proj(embed)

        return TransformerModel(
            frontend,
            encoder,
            frontend,
            decoder,
            final_proj,
            max_target_seq_len=config.max_seq_len,
            target_vocab_info=config.vocab_info,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(
            num_embeddings=config.vocab_info.size,
            embedding_dim=config.model_dim,
            pad_idx=config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def create_frontend(self, embed: Embedding) -> TransformerFrontend:
        config = self._config

        pos_encoder = self.create_position_encoder()

        return TransformerEmbeddingFrontend(
            embed, pos_encoder, dropout_p=config.dropout_p
        )

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        return SinusoidalPositionEncoder(
            config.model_dim, config.max_seq_len, _legacy_pad_idx=1
        )

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_encoder_layer()

            layers.append(layer)

        return StandardTransformerEncoder(layers, norm_order=config.norm_order)

    def create_encoder_layer(self) -> TransformerEncoderLayer:
        config = self._config

        self_attn = self.create_attention(config.num_encoder_attn_heads)

        ffn = self.create_ffn()

        return StandardTransformerEncoderLayer(
            self_attn, ffn, dropout_p=config.dropout_p, norm_order=config.norm_order
        )

    def create_attention(self, num_heads: int) -> MultiheadAttention:
        config = self._config

        sdpa = create_default_sdpa(attn_dropout_p=config.dropout_p)

        return StandardMultiheadAttention(config.model_dim, num_heads, sdpa=sdpa)

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        return StandardFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=True,
            norm_order=config.norm_order,
        )

    def create_decoder(self) -> TransformerDecoder:
        config = self._config

        layers = []

        for _ in range(config.num_decoder_layers):
            layer = self.create_decoder_layer()

            layers.append(layer)

        return StandardTransformerDecoder(layers, norm_order=config.norm_order)

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        config = self._config

        self_attn = self.create_attention(config.num_decoder_attn_heads)

        encoder_decoder_attn = self.create_attention(config.num_decoder_attn_heads)

        ffn = self.create_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=config.dropout_p,
            norm_order=config.norm_order,
        )

    def create_final_proj(self, embed: Embedding) -> Projection:
        config = self._config

        if isinstance(embed, StandardEmbedding):
            return TiedProjection(embed.weight, bias=None)

        return Linear(config.model_dim, config.vocab_info.size, bias=False)


def init_final_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, std=proj.input_dim**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)

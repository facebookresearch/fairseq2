# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from functools import partial

import torch
from torch import Tensor

from fairseq2.models.qwen25._config import Qwen25Config, Qwen25HFRopeConfigPatch
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    init_final_projection,
)
from fairseq2.models.transformer_decoder import TransformerDecoderModel
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    RMSNorm,
    ReferenceRotaryEncoder,
    StandardEmbedding,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    GLUFeedForwardNetworkV2,
    MultiheadAttention,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa
)
from fairseq2.nn.transformer._multihead_attention import init_output_projection
from fairseq2.typing import DataType, Device

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

def create_qwen25_model(config: Qwen25Config) -> TransformerDecoderModel:
    return Qwen25Factory(config).create_model()

class Qwen25Factory:
    _config: Qwen25Config

    def __init__(self, config: Qwen25Config) -> None:
        self._config = config

    def create_model(self) -> TransformerDecoderModel:
        config = self._config

        decoder_frontend = self.create_decoder_frontend()

        decoder = self.create_decoder()

        final_proj = self.create_final_proj()

        return TransformerDecoderModel(
            decoder_frontend,
            decoder,
            final_proj,
            max_seq_len=config.max_seq_len,
            vocab_info=config.vocab_info,
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        config = self._config

        embed = self.create_embedding()

        return TransformerEmbeddingFrontend(
            embed, pos_encoder=None, no_scale=True, dropout_p=config.dropout_p
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(
            num_embeddings=config.vocab_info.size, embedding_dim=config.model_dim
        )

    def create_decoder(self) -> TransformerDecoder:
        config = self._config

        pos_encoder = self.create_position_encoder()

        layers = []

        for _ in range(config.num_layers):
            layer = self.create_decoder_layer(pos_encoder)

            layers.append(layer)

        return StandardTransformerDecoder(
            layers,
            dropout_p=config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.create_layer_norm,
        )

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        return ReferenceRotaryEncoder(
            config.model_dim // config.num_attn_heads,
            config.max_seq_len,
            theta=config.rope_theta,
        )

    def create_decoder_layer(
        self, pos_encoder: PositionEncoder
    ) -> TransformerDecoderLayer:
        self_attn = self.create_attention(pos_encoder)

        ffn = self.create_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.create_layer_norm,
        )

    def create_attention(self, pos_encoder: PositionEncoder) -> MultiheadAttention:
        config = self._config

        sdpa = create_default_sdpa(attn_dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            num_key_value_heads=config.num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=pos_encoder,
            bias=True,
            output_proj_bias=False,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        return GLUFeedForwardNetworkV2(
            config.model_dim,
            config.ffn_inner_dim,
            bias=False,
            inner_dropout_p=config.dropout_p,
        )

    def create_final_proj(self) -> Projection:
        config = self._config

        return Linear(
            config.model_dim,
            config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
        )

    @staticmethod
    def create_layer_norm(
        model_dim: int, *, device: Device | None = None, dtype: DataType | None = None
    ) -> LayerNorm:
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype, eps=1e-06)

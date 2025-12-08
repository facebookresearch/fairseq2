# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.models.opt.config import OPTConfig
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    LocalAttentionStateFactory,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    StandardTransformerLMDecoderLayer,
    TransformerLM,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    LearnedPositionEncoder,
    Linear,
    PositionEncoder,
    Projection,
    StandardEmbedding,
    StandardLayerNorm,
)


def create_opt_model(config: OPTConfig) -> TransformerLM:
    return OPTFactory(config).create_model()


class OPTFactory:
    def __init__(self, config: OPTConfig) -> None:
        self._config = config

    def create_model(self) -> TransformerLM:
        config = self._config

        decoder_frontend = self.create_decoder_frontend()

        decoder = self.create_decoder()

        final_proj = self.create_final_projection()

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_seq_len,
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        config = self._config

        embed = self.create_embedding()

        pos_encoder = self.create_position_encoder()

        return TransformerEmbeddingFrontend(
            config.model_dim,
            embed,
            pos_encoder=pos_encoder,
            no_scale=True,
            # dropout_p=config.dropout_p,  # TODO: check if there is dropout here
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(config.vocab_size, config.model_dim, config.pad_idx)

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        layers = []

        for _ in range(config.num_layers):
            layer = self.create_decoder_layer()

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        return LearnedPositionEncoder(
            config.model_dim, config.max_seq_len, _legacy_pad_idx=1
        )

    def create_decoder_layer(self) -> TransformerLMDecoderLayer:
        config = self._config

        self_attn = self.create_self_attention()

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn()

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias(attn_window_len=config.attn_window_len)

        sdpa = create_default_sdpa(attn_bias)

        state_factory = LocalAttentionStateFactory(config.attn_window_len)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            bias=True,
            state_factory=state_factory,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        return StandardFeedForwardNetwork(
            config.model_dim, config.ffn_inner_dim, bias=True
        )

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        return StandardLayerNorm(config.model_dim, bias=True)

    def create_final_projection(self) -> Projection:
        config = self._config

        return Linear(
            config.model_dim,
            config.vocab_size,
            bias=False,
            init_fn=_init_final_projection,
        )


def _init_final_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, std=proj.input_dim**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.qwen.config import QwenConfig
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
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
    ColumnShardedLinear,
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    RMSNorm,
    ShardedEmbedding,
    StandardEmbedding,
    TiedProjection,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


def create_qwen_model(config: QwenConfig) -> TransformerLM:
    gangs = maybe_get_current_gangs()

    return QwenFactory(config, gangs).create_model()


class QwenFactory:
    def __init__(self, config: QwenConfig, gangs: Gangs | None = None) -> None:
        self._config = config
        self._gangs = gangs

    def create_model(self) -> TransformerLM:
        config = self._config

        embed = self.create_embedding()

        decoder_frontend = self.create_decoder_frontend(embed)

        decoder = self.create_decoder()

        final_proj = self.create_final_projection(embed)

        pad_idx = None

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            pad_idx,
            config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        def init_embed(embed: StandardEmbedding) -> None:
            embed_dim = embed.weight.shape[1]

            std = embed_dim**-0.5

            _init_truncated_normal(embed.weight, bias=None, std=std)

        embed = StandardEmbedding(
            config.vocab_size, config.model_dim, init_fn=init_embed
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            return ShardedEmbedding.from_embedding(embed, gangs.tp)

        return embed

    def create_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        config = self._config

        return TransformerEmbeddingFrontend(
            config.model_dim,
            embed,
            pos_encoder=None,
            no_scale=True,
            dropout_p=config.dropout_p,
        )

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        pos_encoder = self.create_position_encoder()

        layers = []

        for idx in range(config.num_layers):
            layer = self.create_decoder_layer(idx, pos_encoder)

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        if config.head_dim is not None:
            encoding_dim = config.head_dim
        else:
            encoding_dim = config.model_dim // config.num_attn_heads

        return ReferenceRotaryEncoder(
            encoding_dim, config.max_seq_len, theta=config.rope_theta
        )

    def create_decoder_layer(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        config = self._config

        self_attn = self.create_self_attention(layer_idx, pos_encoder)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()

        sdpa = create_default_sdpa(attn_bias)

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        if config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = config.model_dim // config.num_attn_heads

        if config.k_norm:
            k_norm = self.create_layer_norm(head_dim)
        else:
            k_norm = None

        if config.q_norm:
            q_norm = self.create_layer_norm(head_dim)
        else:
            q_norm = None

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = input_dim**-0.5

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            head_dim=config.head_dim,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            bias=config.qkv_proj_bias,
            q_norm=q_norm,
            k_norm=k_norm,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_projection,
            output_proj_bias=False,
            gangs=self._gangs,
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = input_dim**-0.5

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        return GLUFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=1.0,
            proj_init_fn=init_projection,
            gangs=self._gangs,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, StandardEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{StandardEmbedding}` when `config.tied_embeddings` is set, but is of type `{type(embed)}` instead."
                )

            return TiedProjection(embed.weight, bias=None)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = input_dim**-0.5

            _init_truncated_normal(proj.weight, proj.bias, std=std)

        final_proj = Linear(
            config.model_dim, config.vocab_size, bias=False, init_fn=init_projection
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            return ColumnShardedLinear.from_linear(final_proj, gangs.tp)

        return final_proj

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        config = self._config

        if dim is None:
            dim = config.model_dim

        return RMSNorm(dim, bias=False, eps=1e-06)

    def get_std_scale_factor(self, layer_idx: int) -> float:
        config = self._config

        return (2 * (config.num_layers + 1)) ** 0.5  # type: ignore[no-any-return]


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    if bias is not None:
        nn.init.zeros_(bias)

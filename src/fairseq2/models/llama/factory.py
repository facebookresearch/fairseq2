# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.llama.config import LLaMAConfig, LLaMARoPEScaleConfig
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
    RotaryEncoder,
    ShardedEmbedding,
    StandardEmbedding,
    TiedProjection,
    VocabShardedEmbedding,
)
from fairseq2.utils.tensor import to_tensor


def create_llama_model(config: LLaMAConfig) -> TransformerLM:
    gangs = maybe_get_current_gangs()

    return LLaMAFactory(config, gangs).create_model()


class LLaMAFactory:
    def __init__(self, config: LLaMAConfig, gangs: Gangs | None = None) -> None:
        self._config = config
        self._gangs = gangs

    def create_model(self) -> TransformerLM:
        config = self._config

        embed = self.create_embedding()

        decoder_frontend = self.create_decoder_frontend(embed)

        decoder = self.create_decoder()

        final_proj = self.create_final_projection(embed)

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            config.pad_idx,
            config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        init_std = config.init_std

        def init_embed(embed: StandardEmbedding) -> None:
            embed_dim = embed.weight.shape[1]

            std = init_std or (embed_dim**-0.5)

            _init_truncated_normal(embed.weight, bias=None, std=std)

        embed = StandardEmbedding(
            config.vocab_size, config.model_dim, config.pad_idx, init_fn=init_embed
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            if not config.shard_embed_dim:
                return VocabShardedEmbedding.from_embedding(embed, gangs.tp)

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

        if config.use_scaled_rope:
            rope_scale = config.rope_scale

            def init_rope_freqs(pos_encoder: RotaryEncoder) -> Tensor:
                return init_llama_rope_freqs(pos_encoder, rope_scale)

            freqs_init_fn = init_rope_freqs
        else:
            freqs_init_fn = None

        encoding_dim = config.model_dim // config.num_attn_heads

        return RotaryEncoder(
            encoding_dim,
            config.max_seq_len,
            theta=config.rope_theta,
            freqs_init_fn=freqs_init_fn,
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

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_projection,
            bias=False,
            gangs=self._gangs,
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)

        return GLUFeedForwardNetwork(
            config.model_dim,
            ffn_inner_dim,
            bias=False,
            inner_dim_scale=config.ffn_inner_dim_scale,
            inner_dim_to_multiple=config.ffn_inner_dim_multiple_of,
            proj_init_fn=init_projection,
            gangs=self._gangs,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, StandardEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{StandardEmbedding}` when `config.tied_embeddings` is `True`, but is of type `{type(embed)}` instead."
                )

            return TiedProjection(embed.weight, bias=None)

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std)

        final_proj = Linear(
            config.model_dim, config.vocab_size, bias=False, init_fn=init_projection
        )

        gangs = self._gangs

        if gangs is not None and gangs.tp.size > 1:
            return ColumnShardedLinear.from_linear(final_proj, gangs.tp)

        return final_proj

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        return RMSNorm(config.model_dim, bias=False)

    def get_std_scale_factor(self, layer_idx: int) -> float:
        config = self._config

        match config.init_std_scale:
            case "layer":
                n = layer_idx
            case "stack":
                n = config.num_layers
            case "none":
                return 1.0
            case _:
                raise ValueError(
                    f"`config.init_std_scale` must be 'none', 'layer', or 'stack', but is '{config.init_std_scale}' instead."
                )

        return (2 * (n + 1)) ** 0.5  # type: ignore[no-any-return]


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    if bias is not None:
        nn.init.zeros_(bias)


def init_llama_rope_freqs(
    pos_encoder: RotaryEncoder, rope_scale: LLaMARoPEScaleConfig
) -> Tensor:
    device = pos_encoder.freqs.device

    # (E / 2)
    indices = torch.arange(
        0, pos_encoder.encoding_dim, step=2, device=device, dtype=torch.float32
    )

    freqs = 1.0 / (pos_encoder.theta ** (indices / pos_encoder.encoding_dim))

    if device.type == "meta":
        return freqs  # type: ignore[no-any-return]

    old_context_len = rope_scale.original_context_length

    scale_factor = rope_scale.factor

    l_freq_factor, h_freq_factor = rope_scale.frequency_factors

    l_freq_wavelen = old_context_len / l_freq_factor
    h_freq_wavelen = old_context_len / h_freq_factor

    new_freqs = []

    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq

        if wavelen < h_freq_wavelen:
            new_freqs.append(freq)

            continue

        if wavelen > l_freq_wavelen:
            new_freqs.append(freq / scale_factor)

            continue

        smooth = (old_context_len / wavelen - l_freq_factor) / (h_freq_factor - l_freq_factor)  # fmt: skip

        new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return to_tensor(new_freqs, dtype=freqs.dtype, device=device)

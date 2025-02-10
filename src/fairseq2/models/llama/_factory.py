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

from fairseq2.models.llama._config import LLaMAConfig, LLaMARopeScalingConfig
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
    RotaryEncoder,
    StandardEmbedding,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


class LLaMAFactory:
    _config: LLaMAConfig

    def __init__(self, config: LLaMAConfig) -> None:
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

        if config.use_scaled_rope:
            freqs_init_fn = partial(
                init_llama_scaled_freqs, rope_scaling=config.rope_scaling
            )
        else:
            freqs_init_fn = None

        return RotaryEncoder(
            config.model_dim // config.num_attn_heads,
            config.max_seq_len,
            theta=config.rope_theta,
            freqs_init_fn=freqs_init_fn,
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
            bias=False,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        config = self._config

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)

        return GLUFeedForwardNetwork(
            config.model_dim,
            ffn_inner_dim,
            bias=False,
            inner_dim_scale=config.ffn_inner_dim_scale,
            inner_dim_to_multiple=config.ffn_inner_dim_to_multiple,
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
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def init_llama_scaled_freqs(
    pos_encoder: RotaryEncoder, rope_scaling: LLaMARopeScalingConfig
) -> Tensor:
    device = pos_encoder.freqs.device

    # (E / 2)
    indices = torch.arange(
        0, pos_encoder.encoding_dim, step=2, device=device, dtype=torch.float32
    )

    freqs = 1.0 / (pos_encoder.theta ** (indices / pos_encoder.encoding_dim))

    if device.type == "meta":
        return freqs  # type: ignore[no-any-return]

    old_context_len = rope_scaling.original_context_length

    scale_factor = rope_scaling.factor

    l_freq_factor, h_freq_factor = rope_scaling.frequency_factors

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

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=device)

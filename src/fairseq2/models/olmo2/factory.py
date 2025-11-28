# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.olmo2.attention import OLMO2MultiheadAttention
from fairseq2.models.olmo2.config import OLMO2Config
from fairseq2.models.olmo2.decoder_layer import OLMO2TransformerLMDecoderLayer
from fairseq2.models.olmo2.normalization import OLMO2RMSNorm
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
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
    ShardedEmbedding,
    StandardEmbedding,
    TiedProjection,
    VocabShardedEmbedding,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


def create_olmo2_model(config: OLMO2Config) -> TransformerLM:
    """Create an OLMO2 model instance."""
    gangs = maybe_get_current_gangs()

    return OLMO2Factory(config, gangs).create_model()


class OLMO2Factory:
    """Factory for creating OLMO2 models.

    OLMO2 is based on LLaMA architecture with the following differences:
    - Olmo2RMSNorm: multiply weight, then convert back to input dtype
    - Add Q/K Norm in attention layers.
    - Use Olmo2 Post-Norm in decoder layer: Attention/FFN -> Norm -> Add Residual
    - Use MHA instead of GQA. OLMO2-32B model use GQA.
    - Use a HuggingFace-style RoPE module for OLMO2 with rotate_half
    """

    def __init__(self, config: OLMO2Config, gangs: Gangs | None = None) -> None:
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

        rope_encoder = self.create_rope_encoder()

        layers = []

        for idx in range(config.num_layers):
            layer = self.create_decoder_layer(idx, rope_encoder)

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_rope_encoder(self) -> PositionEncoder:
        """Create rotary encoder for OLMO2."""
        config = self._config

        head_dim = config.model_dim // config.num_attn_heads

        return ReferenceRotaryEncoder(
            head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )

    def create_decoder_layer(
        self, layer_idx: int, rope_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        """Create decoder layer with OLMO2-specific Post-Norm architecture.

        OLMO2 uses a unique Post-Norm ordering:
        Attention/FFN -> Norm -> Add Residual

        This differs from standard Post-Norm which does:
        Attention/FFN -> Add Residual -> Norm
        """
        config = self._config

        self_attn = self.create_self_attention(layer_idx, rope_encoder)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        return OLMO2TransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(
        self, layer_idx: int, rope_encoder: PositionEncoder
    ) -> MultiheadAttention:
        """Create self-attention layer with Q/K Norm and HuggingFace-style RoPE.

        Compared to LLaMA,
        1) OLMO2 adds Q/K Norm after Q and K projections.
        2) Uses HuggingFace-style RoPE (rotate_half) and keep dtypes of cos and sin.
        """
        config = self._config

        attn_bias = CausalAttentionBias()
        sdpa = create_default_sdpa(attn_bias)

        init_std = config.init_std
        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]
            std = init_std or (input_dim**-0.5)
            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        head_dim = config.model_dim // config.num_attn_heads
        # OLMO2 uses Q/K norm on the full projection dimension (num_heads * head_dim)
        # This differs from Qwen which uses norm on just head_dim
        q_norm = self.create_layer_norm(config.num_attn_heads * head_dim)
        k_norm = self.create_layer_norm(config.num_key_value_heads * head_dim)

        return OLMO2MultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            q_norm=q_norm,
            k_norm=k_norm,
            pos_encoder=rope_encoder,
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

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        """Create OLMO2RMSNorm.
        OLMO2 RMS norm differs from Llama RMS norm in the order of operations:
        - Weight and hidden states are multiplied before converting back to the input dtype.
        """
        config = self._config

        if dim is None:
            dim = config.model_dim

        return OLMO2RMSNorm(
            dim,
            bias=False,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def get_std_scale_factor(self, layer_idx: int) -> float:
        config = self._config

        n: int
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

        return float((2 * (n + 1)) ** 0.5)


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    if bias is not None:
        nn.init.zeros_(bias)

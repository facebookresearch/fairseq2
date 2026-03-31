# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from fairseq2.error import NotSupportedError
from fairseq2.models.qwen.attention import Qwen35Attention
from fairseq2.models.qwen.config import Qwen35Config, Qwen35MoeConfig, QwenConfig
from fairseq2.models.qwen.gated_delta_net import GatedDeltaNet
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
    StandardEmbedding,
    TiedProjection,
    VocabShardedEmbedding,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


def create_qwen_model(config: QwenConfig) -> TransformerLM:
    return QwenFactory(config).create_model()


def create_qwen35_model(config: Qwen35Config) -> TransformerLM:
    return Qwen35Factory(config).create_model()


class QwenFactory:
    def __init__(self, config: QwenConfig) -> None:
        self._config = config

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

        return VocabShardedEmbedding(
            config.vocab_size, config.model_dim, config.pad_idx, init_fn=init_embed
        )

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
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, VocabShardedEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{VocabShardedEmbedding}` when `config.tied_embeddings` is `True`, but is of type `{type(embed)}` instead."
                )

            if embed.tp_gang.size > 1:
                raise NotSupportedError(
                    "Tied embeddings are not supported when tensor parallelism is enabled."
                )

            return TiedProjection(embed.weight, bias=None)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = input_dim**-0.5

            _init_truncated_normal(proj.weight, proj.bias, std=std)

        return ColumnShardedLinear(
            config.model_dim, config.vocab_size, bias=False, init_fn=init_projection
        )

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


# ---------------------------------------------------------------------------
# Qwen 3.5 Factory
# ---------------------------------------------------------------------------


class Qwen35Factory:
    """Factory for Qwen 3.5 dense hybrid models."""

    def __init__(self, config: Qwen35Config) -> None:
        self._config = config
        config.__post_init__()

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

        def init_embed(embed: StandardEmbedding) -> None:
            std = embed.weight.shape[1] ** -0.5
            _init_truncated_normal(embed.weight, bias=None, std=std)

        return VocabShardedEmbedding(
            config.vocab_size, config.model_dim, config.pad_idx, init_fn=init_embed
        )

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

        encoding_dim = int(config.head_dim * config.partial_rotary_factor)

        return ReferenceRotaryEncoder(
            encoding_dim, config.max_seq_len, theta=config.rope_theta
        )

    def create_decoder_layer(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        from fairseq2.models.qwen.decoder_layer import Qwen35DecoderLayer

        config = self._config

        assert config.layer_types is not None
        layer_type = config.layer_types[layer_idx]

        self_attn = None
        linear_attn = None

        if layer_type == "full_attention":
            self_attn = self.create_gated_attention(layer_idx, pos_encoder)
        else:
            linear_attn = self.create_gated_delta_net(layer_idx)

        ffn = self.create_ffn(layer_idx)
        self_attn_layer_norm = self.create_layer_norm()
        ffn_layer_norm = self.create_layer_norm()

        return Qwen35DecoderLayer(
            layer_type,
            self_attn=self_attn,
            linear_attn=linear_attn,
            ffn=ffn,
            self_attn_layer_norm=self_attn_layer_norm,
            ffn_layer_norm=ffn_layer_norm,
        )

    def create_gated_attention(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> Qwen35Attention:
        from fairseq2.models.qwen.attention import Qwen35Attention

        config = self._config

        attn_bias = CausalAttentionBias()
        sdpa = create_default_sdpa(attn_bias)

        q_norm = self.create_layer_norm(config.head_dim)
        k_norm = self.create_layer_norm(config.head_dim)

        return Qwen35Attention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            head_dim=config.head_dim,
            num_key_value_heads=config.num_key_value_heads,
            pos_encoder=pos_encoder,
            q_norm=q_norm,
            k_norm=k_norm,
        )

    def create_gated_delta_net(self, layer_idx: int) -> GatedDeltaNet:
        from fairseq2.models.qwen.gated_delta_net import GatedDeltaNet

        config = self._config

        return GatedDeltaNet(
            hidden_size=config.model_dim,
            num_k_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        return GLUFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=1.0,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, VocabShardedEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{VocabShardedEmbedding}` when tied_embeddings is True."
                )
            return TiedProjection(embed.weight, bias=None)

        return ColumnShardedLinear(config.model_dim, config.vocab_size, bias=False)

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        config = self._config
        if dim is None:
            dim = config.model_dim
        return RMSNorm(dim, bias=False, eps=1e-06)


def create_qwen35_moe_model(config: Qwen35MoeConfig) -> TransformerLM:
    return Qwen35MoeFactory(config).create_model()


class Qwen35MoeFactory(Qwen35Factory):
    """Factory for Qwen 3.5 MoE hybrid models."""

    _config: Qwen35MoeConfig

    def __init__(self, config: Qwen35MoeConfig) -> None:
        super().__init__(config)
        self._config = config

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        from fairseq2.models.qwen.moe import Qwen35MoeBlock

        config = self._config

        return Qwen35MoeBlock(
            model_dim=config.model_dim,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        )

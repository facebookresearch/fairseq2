# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory for creating NemotronH models.

Builds a TransformerLM with the 3-way hybrid decoder:
- Mamba2 SSM blocks at 'M' positions
- MoE FFN blocks at 'E' positions
- Standard GQA Attention blocks at 'A' positions
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from fairseq2.models.nemotron.config import NemotronHConfig
from fairseq2.models.nemotron.decoder_layer import NemotronHBlock
from fairseq2.models.nemotron.mamba2 import NemotronHMamba2Mixer
from fairseq2.models.nemotron.moe import NemotronHMoE
from fairseq2.models.transformer import (
    CausalAttentionBias,
    MultiheadAttention,
    StandardMultiheadAttention,
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
    RMSNorm,
    StandardEmbedding,
    TiedProjection,
    VocabShardedEmbedding,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


def create_nemotron_h_model(config: NemotronHConfig) -> TransformerLM:
    """Create a NemotronH language model."""
    return NemotronHFactory(config).create_model()


class NemotronHFactory:
    """Factory for building NemotronH models."""

    def __init__(self, config: NemotronHConfig) -> None:
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

        return VocabShardedEmbedding(config.vocab_size, config.model_dim)

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

        layer_types = config.layer_types

        layers: list[TransformerLMDecoderLayer] = []

        for idx in range(config.num_layers):
            block_type = layer_types[idx]
            layer = self.create_decoder_layer(idx, block_type, pos_encoder)
            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        return ReferenceRotaryEncoder(
            config.attn_head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )

    def create_decoder_layer(
        self,
        layer_idx: int,
        block_type: str,
        pos_encoder: PositionEncoder,
    ) -> TransformerLMDecoderLayer:
        config = self._config

        norm = self.create_layer_norm()

        if block_type == "mamba":
            mixer = self.create_mamba2_mixer(layer_idx)
        elif block_type == "attention":
            mixer = self.create_self_attention(layer_idx, pos_encoder)
        elif block_type == "moe":
            mixer = self.create_moe_block(layer_idx)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        return NemotronHBlock(
            block_type=block_type,  # type: ignore[arg-type]
            mixer=mixer,
            norm=norm,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
            rescale_prenorm_residual=config.rescale_prenorm_residual,
        )

    def create_mamba2_mixer(self, layer_idx: int) -> NemotronHMamba2Mixer:
        config = self._config

        return NemotronHMamba2Mixer(
            config.model_dim,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            state_size=config.ssm_state_size,
            n_groups=config.mamba_n_groups,
            conv_kernel=config.conv_kernel,
            chunk_size=config.chunk_size,
            time_step_min=config.time_step_min,
            time_step_max=config.time_step_max,
            use_conv_bias=config.use_conv_bias,
            proj_bias=config.mamba_proj_bias,
            eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )

    def create_self_attention(
        self,
        layer_idx: int,
        pos_encoder: PositionEncoder,
    ) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()
        sdpa = create_default_sdpa(attn_bias)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            head_dim=config.attn_head_dim,
            num_key_value_heads=config.num_key_value_heads,
            bias=False,  # attention_bias = False
            pos_encoder=pos_encoder,
            output_proj_bias=False,
        )

    def create_moe_block(self, layer_idx: int) -> NemotronHMoE:
        config = self._config

        return NemotronHMoE(
            config.model_dim,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
            norm_topk_prob=config.norm_topk_prob,
            bias=False,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, VocabShardedEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{VocabShardedEmbedding}` "
                    f"when `config.tied_embeddings` is `True`, but is of type "
                    f"`{type(embed)}` instead."
                )
            return TiedProjection(embed.weight, bias=None)

        return ColumnShardedLinear(
            config.model_dim, config.vocab_size, bias=False
        )

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        config = self._config

        if dim is None:
            dim = config.model_dim

        return RMSNorm(dim, bias=False, eps=config.rms_norm_eps)


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    if bias is not None:
        nn.init.zeros_(bias)

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

from fairseq2.models.llama import LLaMAFactory
from fairseq2.models.llama.factory import _init_truncated_normal
from fairseq2.models.olmo2.attention import OLMO2MultiheadAttention
from fairseq2.models.olmo2.config import OLMO2Config

from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
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


def create_olmo2_model(config: OLMO2Config) -> TransformerLM:
    """Create an OLMO2 model instance."""
    gangs = maybe_get_current_gangs()

    return OLMO2Factory(config, gangs).create_model()


class OLMO2Factory(LLaMAFactory):
    """Factory for creating OLMO2 models.

    OLMO2 is based on LLaMA architecture with:
    - RMSNorm with learnable weight (rms_norm_eps)
    - Q/K Norm in attention layers
    - Post-Norm architecture
    - Use MHA instead of GQA. Only 32B model use GQA

    Most components are directly reused from LLaMAFactory.
    """

    _config: OLMO2Config

    def __init__(self, config: OLMO2Config, gangs: Gangs | None = None) -> None:
        super().__init__(config, gangs)
        self._config = config

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        """Create RMSNorm with learnable weight.

        OLMO2 uses RMSNorm with learnable weight, unlike OLMO which uses
        parameter-less LayerNorm.

        OLMO2 RMS norm is identical to Llama RMS norm in HF, except for
        # - Weight and hidden states are multiplied before converting back to the input dtype, rather than after.
        not sure if the order will make any difference
        """
        config = self._config

        # diffs the RMS norm and QK norm
        if dim is None:
            dim = config.model_dim

        return RMSNorm(
            dim,
            bias=False,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def create_self_attention(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> MultiheadAttention:
        """Create self-attention layer with Q/K Norm.

        Compared to LLaMA, OLMO2 adds Q/K Norm after Q and K projections.
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

        # Create Q/K Norm - normalize on full projected dimension (before splitting into heads)
        head_dim = config.model_dim // config.num_attn_heads
        q_norm = self.create_layer_norm(config.num_attn_heads * head_dim)
        k_norm = self.create_layer_norm(config.num_key_value_heads * head_dim)

        return OLMO2MultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_projection,
            bias=False,
            q_norm=q_norm,
            k_norm=k_norm,
            gangs=self._gangs,
        )

    def create_decoder_layer(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        """Create decoder layer with Post-Norm architecture.

        OLMO2 uses Post-Norm instead of Pre-Norm. The norm is applied after
        attention/FFN operations rather than before.
        """
        config = self._config

        self_attn = self.create_self_attention(layer_idx, pos_encoder)

        # Post-Norm: norm is applied after attention/FFN
        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.POST,  # Post-Norm architecture
            dropout_p=config.dropout_p,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from functools import partial
from typing_extensions import override

import torch
import torch.nn as nn
from torch import Tensor

from fairseq2.models.llama._config import LLaMARopeScalingConfig
from fairseq2.models.llama._factory import (
    LLaMAFactory,
    init_llama_rope_freqs,
    _init_truncated_normal,
)
from fairseq2.models.llama4._config import LLaMA4DecoderConfig
from fairseq2.models.llama4.model._frontend import LLaMA4DecoderFrontend
from fairseq2.models.llama4.model.moe._moe import MoE
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
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
    TiedProjection,
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
from fairseq2.nn.transformer._attention_mask import AttentionMaskFactory
from fairseq2.typing import DataType, Device


def create_llama4_model(config: LLaMA4DecoderConfig) -> TransformerDecoderModel:
    return LLaMA4Factory(config).create_model()


class LLaMA4Factory(LLaMAFactory):
    _config: LLaMA4DecoderConfig

    def __init__(self, config: LLaMA4DecoderConfig) -> None:
        self._config = config
    
    @override
    def create_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        config = self._config
        
        # TODO: build image embedding here
        vision_embed = None
        
        return LLaMA4DecoderFrontend(
            embed,
            vision_embed,
            pos_encoder=None,
            no_scale=True,
            dropout_p=config.dropout_p,
        )
    
    @override
    def create_decoder(self) -> TransformerDecoder:
        config = self._config

        pos_encoder = self.create_position_encoder()

        layers = []
        use_local_attn_mask = []

        for idx in range(config.num_layers):
            layer = self.create_decoder_layer(idx, pos_encoder)

            layers.append(layer)
            
            use_local_attn_mask.append(not self._is_nope_layer(idx))

        return StandardTransformerDecoder(
            layers,
            dropout_p=config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.create_layer_norm,
            attention_chunk_size=config.attention_chunk_size,
            use_local_attn_mask=use_local_attn_mask,
        )
    
    @override
    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        std_scale_factor = self._get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)
        
        # TODO: implement interleave_moe_layer_step for Llama 4 Maverick
        
        return MoE(
            config.model_dim,
            ffn_inner_dim,
            config.experts.use_shared_expert,
            config.experts.num_experts,
            config.experts.capacity_factor,
            config.experts.top_k,
            inner_dim_scale=config.ffn_inner_dim_scale,
            inner_dim_to_multiple=config.ffn_inner_dim_to_multiple,
            eval_with_saved_stats=config.experts.eval_with_saved_stats,
            expert_act_threshold=config.experts.expert_act_threshold,
        )

        # return GLUFeedForwardNetwork(
        #     config.model_dim,
        #     ffn_inner_dim,
        #     bias=False,
        #     inner_dim_scale=config.ffn_inner_dim_scale,
        #     inner_dim_to_multiple=config.ffn_inner_dim_to_multiple,
        #     inner_dropout_p=config.dropout_p,
        #     proj_init_fn=init_projection,
        # )

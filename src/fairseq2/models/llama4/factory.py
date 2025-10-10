# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing_extensions import override

from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.llama import LLaMAFactory
from fairseq2.models.llama4.config import Llama4Config
from fairseq2.models.llama4.moe import MoE
from fairseq2.models.llama.factory import _init_truncated_normal
from fairseq2.models.transformer import (
    AttentionBias,
    CausalAttentionBias,
    ChunkedAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import (
    LayerNorm,
    Linear,
    PositionEncoder,
    RMSNorm,
)


def create_llama4_model(config: Llama4Config) -> TransformerLM:
    gangs = maybe_get_current_gangs()

    return Llama4Factory(config, gangs).create_model()


class Llama4Factory(LLaMAFactory):
    _config: Llama4Config

    def __init__(self, config: Llama4Config, gangs: Gangs | None = None) -> None:
        super().__init__(config, gangs=None)
        # TODO(mgleize): for now we zero out _gangs here,
        # until the L4 sharder is updated to the new sharding API.
        self._config = config

    def create_self_attention(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> MultiheadAttention:
        """
        Compared to Llama 1-3, Llama 4 interleaves layers with chunked local attention,
        with RoPE. Global attention layers use NoPE.
        """
        config = self._config

        is_nope_layer = self._is_nope_layer(layer_idx)
        # The iRoPE architecture uses global attention mask for NoPE layers or
        # if chunked local attention is not used
        use_local_attn_mask = not is_nope_layer and config.attention_chunk_size

        attn_bias: AttentionBias
        if use_local_attn_mask:
            attn_bias = ChunkedAttentionBias(
                attn_chunk_size=config.attention_chunk_size
            )
        else:
            attn_bias = CausalAttentionBias()

        sdpa = create_default_sdpa(attn_bias)

        pos_encoder_maybe = pos_encoder if not is_nope_layer else None

        qk_norm = None
        if config.use_qk_norm and not is_nope_layer:
            head_dim = config.model_dim // config.num_attn_heads
            qk_norm = self.create_qk_norm(head_dim)

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
            pos_encoder=pos_encoder_maybe,
            output_proj_init_fn=init_projection,
            bias=False,
            q_norm=qk_norm,
            k_norm=qk_norm,
        )

    def _is_nope_layer(self, layer_idx: int) -> bool:
        return (
            self._config.nope_layer_interval is not None
            and (layer_idx + 1) % self._config.nope_layer_interval == 0
        )

    @override
    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)

        if (layer_idx + 1) % config.experts.interleave_moe_layer_step == 0:
            return MoE(
                config.model_dim,
                ffn_inner_dim,
                config.experts.use_shared_expert,
                config.experts.num_experts,
                config.experts.capacity_factor,
                config.experts.top_k,
                inner_dim_scale=config.ffn_inner_dim_scale,
                inner_dim_to_multiple=config.ffn_inner_dim_multiple_of,
            )
        else:
            return GLUFeedForwardNetwork(
                config.model_dim,
                ffn_inner_dim,
                bias=False,
                inner_dim_scale=config.ffn_inner_dim_scale,
                inner_dim_to_multiple=config.ffn_inner_dim_multiple_of,
                inner_dropout_p=config.dropout_p,
                proj_init_fn=init_projection,
            )

    def create_layer_norm(self) -> LayerNorm:
        return RMSNorm(
            self._config.model_dim,
            bias=False,
            elementwise_affine=True,
        )

    def create_qk_norm(self, dim: int) -> LayerNorm:
        return RMSNorm(
            dim,
            bias=False,
            elementwise_affine=False,
        )

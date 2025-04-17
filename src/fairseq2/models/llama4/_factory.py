# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing_extensions import override

from fairseq2.models.llama4._config import LLaMA4DecoderConfig
from fairseq2.models.llama4.model._frontend import LLaMA4DecoderFrontend
from fairseq2.models.llama4.model.moe._moe import MoE
from fairseq2.models.llama4.model.vision._embedding import VisionEmbeddings
from fairseq2.models.llama._factory import (
    LLaMAFactory,
    _init_truncated_normal,
)
from fairseq2.models.transformer import (
    TransformerFrontend,
)
from fairseq2.models.transformer_decoder import TransformerDecoderModel
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    StandardTransformerDecoder,
    TransformerDecoder,
    TransformerNormOrder,
)
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

        if config.vision_config:
            vision_embed = VisionEmbeddings(config.vision_config)
            vision_proj = Linear(
                config.vision_config.output_dim,
                config.model_dim,
                bias=False,
                init_fn=lambda x: None,
            )
        else:
            vision_embed = None
            vision_proj = None

        return LLaMA4DecoderFrontend(
            embed,
            vision_embed,
            vision_proj,
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

        # TODO: re-introduce for Maverick
        # return GLUFeedForwardNetwork(
        #     config.model_dim,
        #     ffn_inner_dim,
        #     bias=False,
        #     inner_dim_scale=config.ffn_inner_dim_scale,
        #     inner_dim_to_multiple=config.ffn_inner_dim_to_multiple,
        #     inner_dropout_p=config.dropout_p,
        #     proj_init_fn=init_projection,
        # )

    @staticmethod
    def create_llama4_rms_norm(
        model_dim: int,
        *,
        elementwise_affine: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        # This is llama-stack version, we can try switching
        return RMSNorm(
            model_dim,
            bias=False,
            impl="py",
            elementwise_affine=elementwise_affine,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def create_layer_norm(
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        return LLaMA4Factory.create_llama4_rms_norm(
            model_dim,
            elementwise_affine=True,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def create_qk_norm(
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        return LLaMA4Factory.create_llama4_rms_norm(
            model_dim,
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )

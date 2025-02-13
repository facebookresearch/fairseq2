# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch.nn import Module
from typing_extensions import override

from fairseq2.models import AbstractModelHandler
from fairseq2.models.s2t_transformer._config import (
    S2T_TRANSFORMER_MODEL_FAMILY,
    S2TTransformerConfig,
)
from fairseq2.models.s2t_transformer._factory import S2TTransformerFactory
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint


class S2TTransformerModelHandler(AbstractModelHandler):
    @property
    @override
    def family(self) -> str:
        return S2T_TRANSFORMER_MODEL_FAMILY

    @property
    @override
    def kls(self) -> type[Module]:
        return TransformerModel

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(S2TTransformerConfig, config)

        return S2TTransformerFactory(config).create_model()

    @override
    def _convert_checkpoint(
        self, checkpoint: dict[str, object], config: object
    ) -> dict[str, object]:
        return convert_s2t_transformer_checkpoint(checkpoint)


def convert_s2t_transformer_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    key_map = {
        # fmt: off
        r"^encoder\.subsample\.conv_layers\.([0-9]+)\.":                   r"encoder_frontend.feature_extractor.layers.\1.conv.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn_layer_norm\.": r"encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.out_proj\.":  r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.":            r"encoder.layers.\1.self_attn.",
        r"^encoder\.transformer_layers\.([0-9]+)\.final_layer_norm\.":     r"encoder.layers.\1.ffn_layer_norm.",
        r"^encoder\.transformer_layers\.([0-9]+)\.fc1\.":                  r"encoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.transformer_layers\.([0-9]+)\.fc2\.":                  r"encoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":           r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":           r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":              r"decoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":                     r"decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.":          r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                              r"decoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                              r"decoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":                 r"decoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.embed_tokens\.":                                       r"decoder_frontend.embed.",
        r"^decoder\.output_projection\.":                                  r"final_proj.",

        # S2T Conformer
        r"^encoder\.linear\.":                                                   r"encoder_frontend.proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.layers.\1.ffn\2_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.layers.\1.ffn\2.inner_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.layers.\1.ffn\2.output_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_q\.":          r"encoder.layers.\1.self_attn.q_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_k\.":          r"encoder.layers.\1.self_attn.k_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_v\.":          r"encoder.layers.\1.self_attn.v_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_out\.":        r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.layers.\1.conv_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.layers.\1.conv.pointwise_conv1.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.layers.\1.conv.depthwise_conv.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.layers.\1.conv.batch_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.layers.\1.conv.pointwise_conv2.",
        r"^encoder\.conformer_layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.layers.\1.layer_norm.",

        # S2T Conformer - RelPos
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_u":   r"encoder.layers.\1.self_attn.sdpa.u_bias",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_v":   r"encoder.layers.\1.self_attn.sdpa.v_bias",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_pos\.": r"encoder.layers.\1.self_attn.sdpa.r_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)

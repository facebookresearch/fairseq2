# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model
from fairseq2.models.s2t_transformer.factory import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
    create_s2t_transformer_model,
    s2t_transformer_archs,
)
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

load_s2t_transformer_config = StandardModelConfigLoader(
    S2T_TRANSFORMER_FAMILY, S2TTransformerConfig, s2t_transformer_archs
)


def convert_s2t_transformer_checkpoint(
    checkpoint: dict[str, Any], config: S2TTransformerConfig
) -> dict[str, Any]:
    """Convert a fairseq S2T Transformer checkpoint to fairseq2 format."""
    try:
        state_dict = checkpoint["model"]
    except KeyError:
        return checkpoint

    # Check if we have a fairseq2 checkpoint.
    if "decoder.output_projection.weight" not in state_dict:
        return checkpoint

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


load_s2t_transformer_model = StandardModelLoader(
    config_loader=load_s2t_transformer_config,
    factory=create_s2t_transformer_model,
    checkpoint_converter=convert_s2t_transformer_checkpoint,
    restrict_checkpoints=False,
)

load_model.register(S2T_TRANSFORMER_FAMILY, load_s2t_transformer_model)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.models.s2t_transformer.config import S2TTransformerConfig
from fairseq2.models.utils.checkpoint import convert_fairseq_state_dict


def convert_s2t_transformer_state_dict(
    state_dict: dict[str, object], config: S2TTransformerConfig
) -> dict[str, object]:
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    if "decoder.embed_tokens.weight" in state_dict:  # fairseq
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
            # fmt: on
        }

        return convert_fairseq_state_dict(state_dict, key_map)

    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

    return state_dict

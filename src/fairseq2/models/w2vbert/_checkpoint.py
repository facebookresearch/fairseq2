# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.models.w2vbert._config import W2VBertConfig
from fairseq2.typing import CPU


def convert_w2vbert_checkpoint(
    checkpoint: dict[str, object], config: W2VBertConfig
) -> dict[str, object]:
    try:
        state_dict = cast(dict[str, Tensor], checkpoint["model"])
    except KeyError:
        return checkpoint

    # Check if we have a fairseq2 checkpoint.
    if "mlm_proj.weight" not in state_dict:
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

        return checkpoint

    state_dict["w2v2_model.quantizer.num_updates"] = torch.zeros((), device=CPU)

    key_map = {
        # fmt: off
        r"^encoder\.pos_conv\.0\.":                                    r"w2v2_model.encoder_frontend.pos_encoder.conv.",
        r"^layer_norm\.":                                              r"w2v2_model.encoder_frontend.post_extract_layer_norm.",
        r"^mask_emb":                                                  r"w2v2_model.masker.temporal_mask_embed",
        r"^post_extract_proj\.":                                       r"w2v2_model.encoder_frontend.model_dim_proj.",
        r"^quantizer\.vars":                                           r"w2v2_model.quantizer.entries",
        r"^quantizer\.weight_proj\.":                                  r"w2v2_model.quantizer.entry_proj.",
        r"^final_proj\.":                                              r"w2v2_model.final_proj.",
        r"^project_q\.":                                               r"w2v2_model.final_target_proj.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"w2v2_model.encoder.layers.\1.conv.batch_norm.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"w2v2_model.encoder.layers.\1.conv.depthwise_conv.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"w2v2_model.encoder.layers.\1.conv_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"w2v2_model.encoder.layers.\1.conv.pointwise_conv1.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"w2v2_model.encoder.layers.\1.conv.pointwise_conv2.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"w2v2_model.encoder.layers.\1.ffn\2_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"w2v2_model.encoder.layers.\1.ffn\2.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"w2v2_model.encoder.layers.\1.ffn\2.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"w2v2_model.encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"w2v2_model.encoder.layers.\1.self_attn.q_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"w2v2_model.encoder.layers.\1.self_attn.k_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"w2v2_model.encoder.layers.\1.self_attn.v_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"w2v2_model.encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"w2v2_model.encoder.layers.\1.self_attn.sdpa.r_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"w2v2_model.encoder.layers.\1.self_attn.sdpa.u_bias",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"w2v2_model.encoder.layers.\1.self_attn.sdpa.v_bias",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"w2v2_model.encoder.layers.\1.layer_norm.",
        r"^encoder\.layer_norm\.":                                     r"w2v2_model.encoder.layer_norm.",
        r"^mlm_proj\.":                                                r"final_bert_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)

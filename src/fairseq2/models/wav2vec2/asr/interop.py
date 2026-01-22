# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.utils.checkpoint import convert_fairseq_state_dict
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig


def convert_wav2vec2_asr_state_dict(
    state_dict: dict[str, object], config: Wav2Vec2AsrConfig
) -> dict[str, object]:

    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    if "w2v_encoder.proj.weight" in state_dict:  # fairseq
        if config.encoder_config.norm_order == TransformerNormOrder.POST:
            # fmt: off
            state_dict["encoder_frontend.layer_norm.weight"] = state_dict["w2v_encoder.w2v_model.encoder.layer_norm.weight"]
            state_dict["encoder_frontend.layer_norm.bias"]   = state_dict["w2v_encoder.w2v_model.encoder.layer_norm.bias"]
            # fmt: on

            del state_dict["w2v_encoder.w2v_model.encoder.layer_norm.weight"]
            del state_dict["w2v_encoder.w2v_model.encoder.layer_norm.bias"]

        key_map = {
            # fmt: off
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"encoder.layers.\1.self_attn_layer_norm.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":    r"encoder.layers.\1.self_attn.q_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":    r"encoder.layers.\1.self_attn.k_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":    r"encoder.layers.\1.self_attn.v_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":  r"encoder.layers.\1.self_attn.output_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.fc1\.":                  r"encoder.layers.\1.ffn.inner_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.fc2\.":                  r"encoder.layers.\1.ffn.output_proj.",
            r"^w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.":     r"encoder.layers.\1.ffn_layer_norm.",
            r"^w2v_encoder\.w2v_model\.encoder\.layer_norm\.":                             r"encoder.layer_norm.",
            r"^w2v_encoder\.w2v_model\.encoder\.embed_tokens\.":                           r"encoder_frontend.embed.",
            r"^w2v_encoder\.w2v_model\.encoder\.pos_conv\.0\.":                            r"encoder_frontend.pos_encoder.conv.",
            r"^w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.([0-9]+)\.0\.":     r"encoder_frontend.feature_extractor.layers.\1.conv.",
            r"^w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.([0-9]+)\.2\.1\.":  r"encoder_frontend.feature_extractor.layers.\1.layer_norm.",
            r"^w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.0\.2\.":            r"encoder_frontend.feature_extractor.layers.0.group_norm.",
            r"^w2v_encoder\.w2v_model\.layer_norm\.":                                      r"encoder_frontend.post_extract_layer_norm.",
            r"^w2v_encoder\.w2v_model\.post_extract_proj\.":                               r"encoder_frontend.model_dim_proj.",
            r"^w2v_encoder\.w2v_model\.mask_emb":                                          r"masker.temporal_mask_embed",
            r"^w2v_encoder\.proj\.":                                                       r"final_proj.",
            # fmt: on
        }

        return convert_fairseq_state_dict(state_dict, key_map)

    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

    return state_dict

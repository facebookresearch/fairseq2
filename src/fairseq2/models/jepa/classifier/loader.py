# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from typing import Any, cast

import torch

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.jepa.classifier import JepaClassifierConfig, create_jepa_classifier_model, jepa_classifier_archs, JEPA_CLASSIFIER_FAMILY
from fairseq2.models.jepa import JEPA_FAMILY
from fairseq2.models.loader import StandardModelLoader
from fairseq2.models.utils.checkpoint import convert_model_state_dict

load_jepa_classifier_config = StandardModelConfigLoader(JEPA_CLASSIFIER_FAMILY, JepaClassifierConfig, jepa_classifier_archs)


def convert_jepa_classifier_checkpoint(checkpoint: dict[str, Any], config: JepaClassifierConfig) -> dict[str, Any]:
    new_checkpoint = checkpoint["classifier"]
    
    key_map = {
        # fmt: off
        r"^module\.pooler\.query_tokens":                         r"pooler.query_tokens",
        r"^module\.pooler\.cross_attention_block\.norm1\.":       r"pooler.decoder.cross_attn_layer_norm.",
        r"^module\.pooler\.cross_attention_block\.xattn\.q\.":    r"pooler.decoder.cross_attn.q_proj",
        r"^module\.pooler\.cross_attention_block\.xattn\.proj\.": r"pooler.decoder.cross_attn.output_proj",
        r"^module\.pooler\.cross_attention_block\.norm2\.":       r"pooler.decoder.ffn_layer_norm",
        r"^module\.pooler\.cross_attention_block\.mlp\.fc1\.":    r"pooler.decoder.ffn.inner_proj",
        r"^module\.pooler\.cross_attention_block\.mlp\.fc2\.":    r"pooler.decoder.ffn.output_proj",
        r"^module\.linear\.":                                     r"head.",
        # fmt: on
    }
    new_checkpoint = convert_model_state_dict(new_checkpoint, key_map)
    
    kv_weight = new_checkpoint.pop("module.pooler.cross_attention_block.xattn.kv.weight")
    k_proj, v_proj = torch.chunk(kv_weight, 2, dim=0)
    new_checkpoint["pooler.decoder.cross_attn.k_proj.weight"] = k_proj
    new_checkpoint["pooler.decoder.cross_attn.v_proj.weight"] = v_proj
    
    kv_bias = new_checkpoint.pop("module.pooler.cross_attention_block.xattn.kv.bias")
    k_bias, v_bias = torch.chunk(kv_bias, 2, dim=0)
    new_checkpoint["pooler.decoder.cross_attn.k_proj.bias"] = k_bias
    new_checkpoint["pooler.decoder.cross_attn.v_proj.bias"] = v_bias

    new_checkpoint = cast(dict[str, object], new_checkpoint)
    
    return {"model": new_checkpoint}


load_jepa_classifier_model = StandardModelLoader(
    config_loader=load_jepa_classifier_config,
    factory=create_jepa_classifier_model,
    checkpoint_converter=convert_jepa_classifier_checkpoint,
)

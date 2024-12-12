# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.jepa.factory import (
    JEPA_FAMILY,
    JepaConfig,
    create_jepa_model,
    jepa_archs,
)
from fairseq2.models.loader import StandardModelLoader
from fairseq2.models.utils.checkpoint import convert_model_state_dict

load_jepa_config = StandardModelConfigLoader(JEPA_FAMILY, JepaConfig, jepa_archs)


def convert_jepa_checkpoint(
    checkpoint: dict[str, Any], config: JepaConfig
) -> dict[str, Any]:
    checkpoint = checkpoint["encoder"]

    del checkpoint["module.backbone.pos_embed"]

    new_checkpoint = {}

    for name, param in checkpoint.items():
        if name.endswith("qkv.weight"):
            q_proj, k_proj, v_proj = torch.chunk(param, 3, dim=0)

            new_checkpoint[name[:-10] + "q_proj.weight"] = q_proj
            new_checkpoint[name[:-10] + "k_proj.weight"] = k_proj
            new_checkpoint[name[:-10] + "v_proj.weight"] = v_proj

            continue

        if name.endswith("qkv.bias"):
            q_bias, k_bias, v_bias = torch.chunk(param, 3, dim=0)

            new_checkpoint[name[:-8] + "q_proj.bias"] = q_bias
            new_checkpoint[name[:-8] + "k_proj.bias"] = k_bias
            new_checkpoint[name[:-8] + "v_proj.bias"] = v_bias

            continue

        new_checkpoint[name] = param

    key_map = {
        # fmt: off
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.q_proj\.":   r"encoder.layers.\1.self_attn.q_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.k_proj\.":   r"encoder.layers.\1.self_attn.k_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.v_proj\.":   r"encoder.layers.\1.self_attn.v_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.norm1\.":          r"encoder.layers.\1.self_attn_layer_norm.",
        r"^module\.backbone\.blocks\.([0-9]+)\.mlp\.fc1\.":       r"encoder.layers.\1.ffn.inner_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.mlp\.fc2\.":       r"encoder.layers.\1.ffn.output_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.norm2\.":          r"encoder.layers.\1.ffn_layer_norm.",
        r"^module\.backbone\.norm\.":                             r"encoder.layer_norm.",
        r"^module\.backbone\.patch_embed\.proj\.":                r"encoder_frontend.feature_extractor.conv.",
        # fmt: on
    }

    checkpoint = convert_model_state_dict(new_checkpoint, key_map)

    return {"model": checkpoint}


load_jepa_model = StandardModelLoader(
    config_loader=load_jepa_config,
    factory=create_jepa_model,
    checkpoint_converter=convert_jepa_checkpoint,
)

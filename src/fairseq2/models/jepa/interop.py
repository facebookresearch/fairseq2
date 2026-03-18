# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from fairseq2.models.jepa.config import JepaConfig
from fairseq2.models.utils.checkpoint import convert_state_dict


def convert_jepa_state_dict(
    state_dict: dict[str, object], config: JepaConfig
) -> dict[str, object]:
    encoder_state_dict = state_dict.get("target_encoder")
    if encoder_state_dict is None:
        encoder_state_dict = state_dict.get("encoder")
        if encoder_state_dict is None:
            raise ValueError(
                "`state_dict` does contain neither a 'target_encoder' nor an 'encoder' key."
            )

    if not isinstance(encoder_state_dict, dict):
        raise TypeError(
            f"The encoder state in `state_dict` is expected to be of type `{dict}`, but is of type `{type(encoder_state_dict)}` instead."
        )

    return _convert_jepa_encoder_state_dict(encoder_state_dict)


def _convert_jepa_encoder_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    try:
        del state_dict["module.backbone.pos_embed"]
    except KeyError:
        pass

    new_state_dict: dict[str, object] = {}

    for name, param in state_dict.items():
        if not isinstance(param, Tensor):
            raise TypeError(
                f"`state_dict['encoder'][{name}]` is expected to be of type `{Tensor}`, but is of type `{type(param)}` instead."
            )

        if name.endswith("qkv.weight"):
            q_proj, k_proj, v_proj = torch.chunk(param, 3, dim=0)

            new_state_dict[name[:-10] + "q_proj.weight"] = q_proj
            new_state_dict[name[:-10] + "k_proj.weight"] = k_proj
            new_state_dict[name[:-10] + "v_proj.weight"] = v_proj

            continue

        if name.endswith("qkv.bias"):
            q_bias, k_bias, v_bias = torch.chunk(param, 3, dim=0)

            new_state_dict[name[:-8] + "q_proj.bias"] = q_bias
            new_state_dict[name[:-8] + "k_proj.bias"] = k_bias
            new_state_dict[name[:-8] + "v_proj.bias"] = v_bias

            continue

        new_state_dict[name] = param

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

    return convert_state_dict(new_state_dict, key_map)

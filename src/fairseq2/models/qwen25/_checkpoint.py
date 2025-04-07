# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch import Tensor

from fairseq2.models.qwen25._config import Qwen25Config
from fairseq2.models.utils.checkpoint import convert_model_state_dict
import torch


def convert_qwen_checkpoint(
    checkpoint: dict[str, object], config: Qwen25Config
) -> dict[str, object]:
    # Check if we have a reference or Hugging Face checkpoint.
    head_dim = config.model_dim // config.num_attn_heads

    def permute_rotary(w: Tensor, num_heads: int) -> Tensor:
        # (H, M) -> (H_d, 2, D / 2, M)
        w = w.view(num_heads, 2, head_dim // 2, config.model_dim)

        # (H_d, 2, D / 2, M) -> (H_d, D / 2, 2, M)
        w = w.transpose(1, 2)

        # (H_d, D / 2, 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    for idx in range(config.num_layers):
        q_key = f"model.layers.{idx}.self_attn.q_proj.weight"
        k_key = f"model.layers.{idx}.self_attn.k_proj.weight"

        q_proj = cast(Tensor, checkpoint[q_key])
        k_proj = cast(Tensor, checkpoint[k_key])

        q_proj = permute_rotary(q_proj, config.num_attn_heads)
        k_proj = permute_rotary(k_proj, config.num_key_value_heads)

        checkpoint[q_key] = q_proj
        checkpoint[k_key] = k_proj

    key_map = {
        # fmt: off
        r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
        r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
        r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
        r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.output_proj.",
        r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_layer_norm.",
        r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"decoder.layers.\1.ffn.gate_proj.",
        r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"decoder.layers.\1.ffn.output_proj.",
        r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"decoder.layers.\1.ffn.inner_proj.",
        r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"decoder.layers.\1.self_attn_layer_norm.",
        r"^model\.norm\.":                                       r"decoder.layer_norm.",
        r"^model\.embed_tokens\.":                               r"decoder_frontend.embed.",
        r"^lm_head\.":                                           r"final_proj.",
        # fmt: on
    }

    checkpoint = convert_model_state_dict(checkpoint, key_map)

    # # adding head scaler
    # head_dim = config.model_dim // config.num_attn_heads
    # scale = head_dim**-0.5
    # scale_tensor = torch.empty(config.num_attn_heads).fill_(scale) 
    # for idx in range(config.num_layers):
    #     key = f"decoder.layers.{idx}.self_attn.head_scale_weight"
    #     checkpoint[key] = scale_tensor

    return {"model": checkpoint}

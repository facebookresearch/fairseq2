# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch import Tensor

from fairseq2.models.qwen._config import QwenConfig
from fairseq2.models.utils.checkpoint import convert_model_state_dict
import torch

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

reverse_key_map = {
    # fmt: off
    r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":          r"model.layers.\1.self_attn.q_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":          r"model.layers.\1.self_attn.k_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":          r"model.layers.\1.self_attn.v_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.output_proj\.":     r"model.layers.\1.self_attn.o_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":             r"model.layers.\1.post_attention_layernorm.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.gate_proj\.":             r"model.layers.\1.mlp.gate_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.output_proj\.":           r"model.layers.\1.mlp.down_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.inner_proj\.":            r"model.layers.\1.mlp.up_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":       r"model.layers.\1.input_layernorm.",
    r"^decoder\.layer_norm\.":                                   r"model.norm.",
    r"^decoder_frontend\.embed\.":                               r"model.embed_tokens.",
    r"^final_proj\.":                                            r"lm_head.",
    # fmt: on
}


def convert_qwen_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:

    checkpoint = convert_model_state_dict(checkpoint, key_map)

    # if weights are tied, we need to create a copy in statedict here for model loading
    if config.tie_embeddings:
        checkpoint["final_proj.weight"] = checkpoint["decoder_frontend.embed.weight"]

    return {"model": checkpoint}


def convert_qwen_fs2_to_hf_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
):

    checkpoint = convert_model_state_dict(checkpoint, reverse_key_map)

    # if emb weights are tied, we need to remove the lm head from ckpt
    if config.tie_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.utils.checkpoint import convert_checkpoint

# isort: split

from fairseq2.models.qwen._config import QwenConfig


def export_qwen_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> tuple[dict[str, object], dict[str, object]]:
    hg_config = _convert_config(config)

    hg_checkpoint = _convert_checkpoint(checkpoint, config)

    return hg_checkpoint, hg_config


def _convert_config(config: QwenConfig) -> dict[str, object]:
    return {
        "hidden_size": config.model_dim,
        "max_position_embeddings": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tied_embeddings,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attn_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.ffn_inner_dim,
        "rope_theta": config.rope_theta,
    }


def _convert_checkpoint(
    checkpoint: dict[str, object], config: QwenConfig
) -> dict[str, object]:
    key_map = {
        # fmt: off
        r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":      r"model.layers.\1.self_attn.q_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":      r"model.layers.\1.self_attn.k_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":      r"model.layers.\1.self_attn.v_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.output_proj\.": r"model.layers.\1.self_attn.o_proj.",
        r"^decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":         r"model.layers.\1.post_attention_layernorm.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.gate_proj\.":         r"model.layers.\1.mlp.gate_proj.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.output_proj\.":       r"model.layers.\1.mlp.down_proj.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.inner_proj\.":        r"model.layers.\1.mlp.up_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":   r"model.layers.\1.input_layernorm.",
        r"^decoder\.layer_norm\.":                               r"model.norm.",
        r"^decoder_frontend\.embed\.":                           r"model.embed_tokens.",
        r"^final_proj\.":                                        r"lm_head.",
        # fmt: on
    }

    checkpoint = convert_checkpoint(checkpoint, key_map)

    if config.tied_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint

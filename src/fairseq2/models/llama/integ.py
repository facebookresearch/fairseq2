# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.models.llama.factory import LLaMAConfig
from fairseq2.models.utils.checkpoint import convert_model_state_dict


def get_ffn_dim_multipliers(architecture: str) -> float:
    ffn_dim_multipliers = {
        "llama2_70b": 1.3,
        "llama3_8b": 1.3,
        "llama3_70b": 1.3,
        "llama3_1_8b": 1.3,
        "llama3_1_70b": 1.3,
        "llama3_1_405b": 1.2,
        "llama3_2_1b": 1.5,
    }

    return ffn_dim_multipliers.get(architecture, 1.0)


def convert_to_reference_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Convert a fairseq2 LLaMA checkpoint to the reference format."""
    try:
        model_key = checkpoint["model_key"]
    except KeyError:
        model_key = "model"

    state_dict = checkpoint[model_key]

    key_map = {
        # fmt: off
        r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":      r"layers.\1.attention.wq.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":      r"layers.\1.attention.wk.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":      r"layers.\1.attention.wv.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.output_proj\.": r"layers.\1.attention.wo.",
        r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":   r"layers.\1.attention_norm.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.gate_proj\.":         r"layers.\1.feed_forward.w1.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.output_proj\.":       r"layers.\1.feed_forward.w2.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.inner_proj\.":        r"layers.\1.feed_forward.w3.",
        r"^decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":         r"layers.\1.ffn_norm.",
        r"^decoder\.layer_norm\.":                               r"norm.",
        r"^decoder_frontend\.embed\.":                           r"tok_embeddings.",
        r"^final_proj\.":                                        r"output.",
        # fmt: on
    }

    return convert_model_state_dict(state_dict, key_map)


def convert_to_huggingface_config(arch: str, config: LLaMAConfig) -> dict[str, Any]:
    """Convert Llama's config to a dict mirroring Huggingface's format"""

    def compute_intermediate_size(
        n: int, ffn_dim_multiplier: float = 1, multiple_of: int = 256
    ) -> int:
        """From: https://github.com/huggingface/transformers/blob/82fcac0a7e40dc6cc5e3121d714b9b16775293ad/src/transformers/models/llama/convert_llama_weights_to_hf.py#L171"""
        return multiple_of * (
            (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of
        )

    def is_llama_3_2(arch: str) -> bool:
        # TODO: this seems too britle
        return "llama3_2_" in arch

    if config.use_scaled_rope:
        rope_scaling = {
            "factor": 32.0 if is_llama_3_2(arch) else 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    else:
        # mgleize: not sure of the json.dump behavior if rope_scaling is None
        rope_scaling = None

    # we only specify the parameters made explicit in the Huggingface converter
    # https://github.com/huggingface/transformers/blob/93aafdc620d39b9ec714ffecf015a085ea221282/src/transformers/models/llama/convert_llama_weights_to_hf.py#L384
    return {
        "architectures": ["Fairseq2LlamaForCausalLM"],
        "bos_token_id": config.vocab_info.bos_idx,
        "eos_token_id": config.vocab_info.eos_idx,
        "hidden_size": config.model_dim,
        "intermediate_size": compute_intermediate_size(
            config.model_dim,
            get_ffn_dim_multipliers(arch),
            config.ffn_inner_dim_to_multiple,
        ),
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.num_attn_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_scaling": rope_scaling,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": is_llama_3_2(arch),
        "vocab_size": config.vocab_info.size,
    }

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from fairseq2.models.llama import LLaMAConfig
from fairseq2.models.utils.checkpoint import convert_model_state_dict


def convert_to_reference_llama_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    """Convert a fairseq2 LLaMA checkpoint to the reference format."""
    model_key = checkpoint.get("model_key", "model")

    if not isinstance(model_key, str):
        raise TypeError(
            f"The 'model_key' entry in `checkpoint` must be of type `str`, but is of type `{type(model_key)}` instead."
        )

    state_dict = cast(dict[str, object], checkpoint[model_key])

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"The '{model_key}' entry in `checkpoint` is expected to be of type `dict`, but is of type `{type(state_dict)}` instead."
        )

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


def convert_to_hg_llama_config(config: LLaMAConfig) -> dict[str, object]:
    """Convert a fairseq2 LLaMA configuration to the HuggingFace format."""
    multiplier = config.ffn_inner_dim_multiplier

    multiple_of = config.ffn_inner_dim_to_multiple

    # Taken from https://github.com/huggingface/transformers/blob/82fcac0a7e40dc6cc5e3121d714b9b16775293ad/src/transformers/models/llama/convert_llama_weights_to_hf.py#L171.
    intermediate_size = multiple_of * ((int(multiplier * int(8 * config.model_dim / 3)) + multiple_of - 1) // multiple_of)  # fmt: skip

    if config.rope_scaling is not None:
        rope_scaling = {
            "factor": config.rope_scaling.factor,
            "low_freq_factor": config.rope_scaling.frequency_factors[0],
            "high_freq_factor": config.rope_scaling.frequency_factors[1],
            "original_max_position_embeddings": config.rope_scaling.original_context_length,
            "rope_type": "llama3",
        }
    else:
        rope_scaling = None

    # We only specify the parameters made explicit in the Hugging Face converter.
    # See https://github.com/huggingface/transformers/blob/93aafdc620d39b9ec714ffecf015a085ea221282/src/transformers/models/llama/convert_llama_weights_to_hf.py#L384.
    return {
        "architectures": ["Fairseq2LlamaForCausalLM"],
        "bos_token_id": config.vocab_info.bos_idx,
        "eos_token_id": config.vocab_info.eos_idx,
        "hidden_size": config.model_dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.num_attn_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_scaling": rope_scaling,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_embeddings,
        "vocab_size": config.vocab_info.size,
    }

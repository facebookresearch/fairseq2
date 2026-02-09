# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final

from fairseq2.models.gemma3n.config import Gemma3nConfig
from fairseq2.models.utils.checkpoint import convert_state_dict

# HuggingFace → fairseq2 key mappings
_HG_KEY_MAP: Final = {
    # fmt: off
    # Embedding layers
    r"^model\.language_model\.embed_tokens\.":                        r"decoder_frontend.embed.",
    r"^lm_head\.":                                                     r"final_proj.",

    # Decoder layers - attention
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.": r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.": r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.": r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.": r"decoder.layers.\1.self_attn.output_proj.",

    # Decoder layers - normalization
    r"^model\.language_model\.layers\.([0-9]+)\.input_layernorm\.":              r"decoder.layers.\1.self_attn_layer_norm.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.":     r"decoder.layers.\1.ffn_layer_norm.",

    # Decoder layers - FFN (standard and AltUp)
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.":   r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.":     r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.":   r"decoder.layers.\1.ffn.output_proj.",

    # Final normalization
    r"^model\.language_model\.norm\.":                                r"decoder.layer_norm.",

    # TODO: LAuReL and PLE (skip for now, will cause mismatches)
    # r"^model\.language_model\.layers\.([0-9]+)\.laurel\.":
    # r"^model\.language_model\.layers\.([0-9]+)\.per_layer_":
    # r"^model\.language_model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":
    # r"^model\.language_model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.":
    # r"^model\.language_model\.layers\.([0-9]+)\.post_per_layer_input_norm\.":
    # fmt: on
}


def convert_gemma3n_state_dict(
    state_dict: dict[str, object],
    config: Gemma3nConfig,
) -> dict[str, object]:
    """Convert a HuggingFace Gemma3n state dictionary to fairseq2 format.

    Args:
        state_dict: The HuggingFace Gemma3n state dictionary.
        config: The Gemma3n configuration.

    Returns:
        The fairseq2-compatible state dictionary.

    Notes:
        This is a stub implementation for Phase 1. Full implementation
        including RoPE permutation, PLE parameters, and AltUp handling
        will be added in Phase 2-3.
    """
    # Note: embed_tokens is in shard 1, layers are in shard 3
    # We don't strictly validate the key since conversion works on subsets

    # TODO(Phase 2): Add RoPE permutation for dual-theta encoding
    # TODO(Phase 3): Handle PLE parameters mapping
    # TODO(Phase 3): Handle AltUp parameters mapping

    return convert_state_dict(state_dict, _HG_KEY_MAP)


def convert_to_hf_gemma3n_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert a fairseq2 Gemma3n state dictionary to HuggingFace format.

    Args:
        state_dict: The fairseq2 Gemma3n state dictionary.

    Returns:
        The HuggingFace-compatible state dictionary.

    Notes:
        This is a stub implementation for Phase 1. Full bidirectional
        conversion will be added in later phases.
    """
    # TODO(Phase 2-3): Implement reverse conversion
    raise NotImplementedError(
        "fairseq2 → HuggingFace conversion not yet implemented."
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, cast

from torch import Tensor

from fairseq2.models.olmo2.config import OLMO2Config
from fairseq2.models.utils.checkpoint import convert_state_dict

_HG_KEY_MAP: Final = {
    # fmt: off
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":                r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":                r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":                r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":                r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":                 r"decoder.layers.\1.self_attn.q_norm.",  # OLMO2 Q/K Norm
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":                 r"decoder.layers.\1.self_attn.k_norm.",  # OLMO2 Q/K Norm
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.":         r"decoder.layers.\1.self_attn_layer_norm.",  # Post-Norm after attention
    r"^model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":      r"decoder.layers.\1.ffn_layer_norm.",  # Post-Norm after FFN
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":                    r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":                    r"decoder.layers.\1.ffn.output_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":                      r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.norm\.":                                                r"decoder.layer_norm.",
    r"^model\.embed_tokens\.":                                        r"decoder_frontend.embed.",
    r"^lm_head\.":                                                   r"final_proj.",
    # fmt: on
}


def convert_olmo2_state_dict(
    state_dict: dict[str, object], config: OLMO2Config
) -> dict[str, object]:
    """Convert OLMO2 state dictionary from HuggingFace format to fairseq2 format.

    OLMO2 uses Post-Norm architecture, so:
    - No input_layernorm (Pre-Norm)
    - Has post_attention_layernorm (after attention)
    - Has post_feedforward_layernorm (after FFN)
    - Has Q/K Norm in attention layers
    """
    # Handle legacy format (wrapped in "model" key)
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    if "model.embed_tokens.weight" in state_dict:  # HuggingFace format
        # Note: OLMO2 may need RoPE weight permutation similar to LLaMA
        # This depends on the actual checkpoint format. For now, we'll check
        # if weights need permutation by comparing shapes.

        # Apply key mapping
        state_dict = convert_state_dict(state_dict, _HG_KEY_MAP)

    elif "tok_embeddings.weight" in state_dict:  # Reference format
        key_map = {
            # fmt: off
            r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
            r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
            r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
            r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.output_proj.",
            r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
            r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
            r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_layer_norm.",
            r"^norm\.":                               r"decoder.layer_norm.",
            r"^tok_embeddings\.":                     r"decoder_frontend.embed.",
            r"^output\.":                             r"final_proj.",
            # fmt: on
        }

        # Filter out pre-computed RoPE frequencies
        state_dict = {k: v for (k, v) in state_dict.items() if "rope.freqs" not in k}

        state_dict = convert_state_dict(state_dict, key_map)

    # Handle tied embeddings
    if config.tied_embeddings:
        state_dict["final_proj.weight"] = state_dict["decoder_frontend.embed.weight"]

    return state_dict

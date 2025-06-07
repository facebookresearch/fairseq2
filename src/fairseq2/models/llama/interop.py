# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, cast

from torch import Tensor

from fairseq2.models.llama.config import LLaMAConfig
from fairseq2.models.utils.checkpoint import convert_state_dict


def convert_to_ref_llama_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert a fairseq2 LLaMA state dictionary to the reference format."""
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

    return convert_state_dict(state_dict, key_map)


_LLAMA_HG_KEY_MAP: Final = {
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


def convert_llama_state_dict(
    state_dict: dict[str, object], config: LLaMAConfig
) -> dict[str, object]:
    try:
        state_dict = cast(dict[str, object], state_dict["model"])  # legacy
    except KeyError:
        pass

    if "model.embed_tokens.weight" in state_dict:  # Hugging Face
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

            q_proj = cast(Tensor, state_dict[q_key])
            k_proj = cast(Tensor, state_dict[k_key])

            q_proj = permute_rotary(q_proj, config.num_attn_heads)
            k_proj = permute_rotary(k_proj, config.num_key_value_heads)

            state_dict[q_key] = q_proj
            state_dict[k_key] = k_proj

        state_dict = convert_state_dict(state_dict, _LLAMA_HG_KEY_MAP)
    elif "tok_embeddings.weight" in state_dict:  # reference
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

        # We do not need the pre-computed 'rope.freqs' buffers.
        state_dict = {k: v for (k, v) in state_dict.items() if "rope.freqs" not in k}

        state_dict = convert_state_dict(state_dict, key_map)

    if config.tied_embeddings:
        state_dict["final_proj.weight"] = state_dict["decoder_frontend.embed.weight"]

    return state_dict

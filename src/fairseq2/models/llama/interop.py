# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, cast

from torch import Tensor

from fairseq2.models.family import HuggingFaceExport
from fairseq2.models.llama.config import LLaMAConfig
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map


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


_HG_KEY_MAP: Final = {
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

        state_dict = convert_state_dict(state_dict, _HG_KEY_MAP)
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


def export_llama(
    state_dict: dict[str, object], config: LLaMAConfig
) -> HuggingFaceExport:
    hg_state_dict = _convert_to_hg_state_dict(state_dict, config)

    hg_config = _convert_to_hg_config(config)

    return HuggingFaceExport(
        hg_state_dict,
        hg_config,
        config_kls_name="LlamaConfig",
        arch="LlamaForCausalLM",
    )


def _convert_to_hg_state_dict(
    state_dict: dict[str, object], config: LLaMAConfig
) -> dict[str, object]:
    head_dim = config.model_dim // config.num_attn_heads

    def permute_rotary(w: Tensor, num_heads: int) -> Tensor:
        # (H, M) -> (H_d, D / 2, 2, M)
        w = w.view(num_heads, head_dim // 2, 2, config.model_dim)

        # (H_d, D / 2, 2, M) -> (H_d, 2, D / 2, m)
        w = w.transpose(1, 2)

        # (H_d, 2, D / 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    for idx in range(config.num_layers):
        q_key = f"decoder.layers.{idx}.self_attn.q_proj.weight"
        k_key = f"decoder.layers.{idx}.self_attn.k_proj.weight"

        q_proj = cast(Tensor, state_dict[q_key])
        k_proj = cast(Tensor, state_dict[k_key])

        q_proj = permute_rotary(q_proj, config.num_attn_heads)
        k_proj = permute_rotary(k_proj, config.num_key_value_heads)

        state_dict[q_key] = q_proj
        state_dict[k_key] = k_proj

    key_map = create_reverse_key_map(_HG_KEY_MAP)

    hg_state_dict = convert_state_dict(state_dict, key_map)

    if config.tied_embeddings:
        del hg_state_dict["lm_head.weight"]

    return hg_state_dict


def _convert_to_hg_config(config: LLaMAConfig) -> dict[str, object]:
    multiplier = config.ffn_inner_dim_multiplier

    multiple_of = config.ffn_inner_dim_multiple_of

    intermediate_size = multiple_of * ((int(multiplier * int(8 * config.model_dim / 3)) + multiple_of - 1) // multiple_of)  # fmt: skip

    if config.use_scaled_rope:
        rope_scale = {
            "factor": config.rope_scale.factor,
            "low_freq_factor": config.rope_scale.frequency_factors[0],
            "high_freq_factor": config.rope_scale.frequency_factors[1],
            "original_max_position_embeddings": config.rope_scale.original_context_length,
            "rope_type": "llama3",
        }
    else:
        rope_scale = None

    if config.vocab_size == 32_000:  # LLaMA 1 and 2
        bos_idx = 1
        eos_idx = 2
    else:
        bos_idx = 128_000
        eos_idx = 128_001

    return {
        "bos_token_id": bos_idx,
        "eos_token_id": eos_idx,
        "hidden_size": config.model_dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.num_attn_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_scaling": rope_scale,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tied_embeddings,
        "vocab_size": config.vocab_size,
        "head_dim": config.model_dim // config.num_attn_heads,
    }

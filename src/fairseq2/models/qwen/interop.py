# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, final

import torch
from typing_extensions import override

from fairseq2.models.hg import HuggingFaceConfig, HuggingFaceConverter
from fairseq2.models.qwen.config import Qwen35Config, Qwen35MoeConfig, QwenConfig
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.utils.config import cast_config_type

_HG_KEY_MAP: Final = {
    # fmt: off
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":        r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":        r"decoder.layers.\1.self_attn.k_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_layer_norm.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"decoder.layers.\1.ffn.output_proj.",
    r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"decoder.layers.\1.self_attn_layer_norm.",
    r"^model\.norm\.":                                       r"decoder.layer_norm.",
    r"^model\.embed_tokens\.":                               r"decoder_frontend.embed.",
    r"^lm_head\.":                                           r"final_proj.",
    # fmt: on
}


def convert_qwen_state_dict(
    state_dict: dict[str, object], config: QwenConfig
) -> dict[str, object]:
    if "model.embed_tokens.weight" in state_dict:  # Hugging Face
        state_dict = convert_state_dict(state_dict, _HG_KEY_MAP)

    if config.tied_embeddings:
        state_dict["final_proj.weight"] = state_dict["decoder_frontend.embed.weight"]

    return state_dict

# HG-side RMSNorm key suffixes for reverse conversion (weight -= 1.0).
_QWEN35_HG_RMSNORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "model.norm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


@final
class _Qwen35HuggingFaceConverter(HuggingFaceConverter):
    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, Qwen35Config)

        data: dict[str, object] = {
            "hidden_size": config.model_dim,
            "max_position_embeddings": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tied_embeddings,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_attn_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.ffn_inner_dim,
            "partial_rotary_factor": config.partial_rotary_factor,
            "rope_theta": config.rope_theta,
            "full_attention_interval": config.full_attention_interval,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
        }

        return HuggingFaceConfig(
            data, kls_name="Qwen3_5TextConfig", arch="Qwen3_5ForCausalLM"
        )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, Qwen35Config)

        key_map = create_reverse_key_map(_QWEN35_HG_KEY_MAP)

        hg_state_dict = convert_state_dict(state_dict, key_map)

        for key in list(hg_state_dict.keys()):
            if any(key.endswith(suffix) for suffix in _QWEN35_HG_RMSNORM_SUFFIXES):
                weight = hg_state_dict[key]
                if isinstance(weight, torch.Tensor):
                    hg_state_dict[key] = weight - 1.0

        if config.tied_embeddings:
            hg_state_dict.pop("lm_head.weight", None)

        return hg_state_dict


@final
class _Qwen35MoeHuggingFaceConverter(HuggingFaceConverter):
    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, Qwen35MoeConfig)

        data: dict[str, object] = {
            "hidden_size": config.model_dim,
            "max_position_embeddings": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tied_embeddings,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_attn_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.ffn_inner_dim,
            "partial_rotary_factor": config.partial_rotary_factor,
            "rope_theta": config.rope_theta,
            "full_attention_interval": config.full_attention_interval,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
            "router_aux_loss_coef": config.router_aux_loss_coef,
        }

        return HuggingFaceConfig(
            data, kls_name="Qwen3_5TextConfig", arch="Qwen3_5MoeForCausalLM"
        )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, Qwen35MoeConfig)

        key_map = create_reverse_key_map(_QWEN35_MOE_HG_KEY_MAP)

        hg_state_dict = convert_state_dict(state_dict, key_map)

        for key in list(hg_state_dict.keys()):
            if any(key.endswith(suffix) for suffix in _QWEN35_HG_RMSNORM_SUFFIXES):
                weight = hg_state_dict[key]
                if isinstance(weight, torch.Tensor):
                    hg_state_dict[key] = weight - 1.0

        if config.tied_embeddings:
            hg_state_dict.pop("lm_head.weight", None)

        return hg_state_dict


@final
class _QwenHuggingFaceConverter(HuggingFaceConverter):
    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, QwenConfig)

        data: dict[str, object] = {
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

        if config.qkv_proj_bias:
            return HuggingFaceConfig(
                data, kls_name="Qwen2Config", arch="Qwen2ForCausalLM"
            )
        else:
            return HuggingFaceConfig(
                data, kls_name="Qwen3Config", arch="Qwen3ForCausalLM"
            )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, QwenConfig)

        key_map = create_reverse_key_map(_HG_KEY_MAP)

        hg_state_dict = convert_state_dict(state_dict, key_map)

        if config.tied_embeddings:
            del hg_state_dict["lm_head.weight"]

        return hg_state_dict


# ---------------------------------------------------------------------------
# Qwen 3.5 interop
# ---------------------------------------------------------------------------

_QWEN35_HG_KEY_MAP: Final = {
    # fmt: off
    # Full attention layers
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":        r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":        r"decoder.layers.\1.self_attn.k_norm.",
    # Linear attention layers (GatedDeltaNet)
    r"^model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.": r"decoder.layers.\1.linear_attn.in_proj_qkv.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.in_proj_z\.":   r"decoder.layers.\1.linear_attn.in_proj_z.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.in_proj_b\.":   r"decoder.layers.\1.linear_attn.in_proj_b.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.in_proj_a\.":   r"decoder.layers.\1.linear_attn.in_proj_a.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.conv1d\.":      r"decoder.layers.\1.linear_attn.conv1d.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.dt_bias":       r"decoder.layers.\1.linear_attn.dt_bias",
    r"^model\.layers\.([0-9]+)\.linear_attn\.A_log":         r"decoder.layers.\1.linear_attn.A_log",
    r"^model\.layers\.([0-9]+)\.linear_attn\.norm\.":        r"decoder.layers.\1.linear_attn.norm.inner_norm.",
    r"^model\.layers\.([0-9]+)\.linear_attn\.out_proj\.":    r"decoder.layers.\1.linear_attn.out_proj.",
    # FFN
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"decoder.layers.\1.ffn.output_proj.",
    # Layer norms
    r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"decoder.layers.\1.self_attn_layer_norm.",
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_layer_norm.",
    # Embeddings
    r"^model\.norm\.":                                       r"decoder.layer_norm.",
    r"^model\.embed_tokens\.":                               r"decoder_frontend.embed.",
    r"^lm_head\.":                                           r"final_proj.",
    # fmt: on
}

# RMSNorm keys that need weight += 1.0 conversion (Qwen 3.5 uses 1+w formula).
# The GatedDeltaNet internal norm (linear_attn.norm) does NOT need conversion.
_QWEN35_RMSNORM_KEYS = (
    "self_attn_layer_norm.weight",
    "ffn_layer_norm.weight",
    "decoder.layer_norm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def convert_qwen35_state_dict(
    state_dict: dict[str, object], config: Qwen35Config
) -> dict[str, object]:
    if "model.embed_tokens.weight" in state_dict:
        state_dict = convert_state_dict(state_dict, _QWEN35_HG_KEY_MAP)

    # Convert (1+w) RMSNorm weights to standard (w) by adding 1.0.
    for key in list(state_dict.keys()):
        if any(key.endswith(suffix) for suffix in _QWEN35_RMSNORM_KEYS):
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor):
                state_dict[key] = weight + 1.0

    if config.tied_embeddings:
        if "decoder_frontend.embed.weight" in state_dict:
            state_dict["final_proj.weight"] = state_dict[
                "decoder_frontend.embed.weight"
            ]
        elif "final_proj.weight" in state_dict:
            state_dict["decoder_frontend.embed.weight"] = state_dict[
                "final_proj.weight"
            ]

    return state_dict


# ---------------------------------------------------------------------------
# Qwen 3.5 MoE interop
# ---------------------------------------------------------------------------

_QWEN35_MOE_HG_KEY_MAP: Final = {
    # fmt: off
    **_QWEN35_HG_KEY_MAP,
    # MoE FFN (replaces dense FFN keys)
    r"^model\.layers\.([0-9]+)\.mlp\.gate\.":                r"decoder.layers.\1.ffn.gate.",
    r"^model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj":  r"decoder.layers.\1.ffn.experts.gate_up_proj",
    r"^model\.layers\.([0-9]+)\.mlp\.experts\.down_proj":     r"decoder.layers.\1.ffn.experts.down_proj",
    r"^model\.layers\.([0-9]+)\.mlp\.shared_expert\.gate_proj\.":  r"decoder.layers.\1.ffn.shared_expert.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.shared_expert\.up_proj\.":    r"decoder.layers.\1.ffn.shared_expert.inner_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.shared_expert\.down_proj\.":  r"decoder.layers.\1.ffn.shared_expert.output_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.shared_expert_gate\.":        r"decoder.layers.\1.ffn.shared_expert_gate.",
    # fmt: on
}

# Remove the dense FFN keys that conflict with MoE keys
for _dense_key in [
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.",
]:
    _QWEN35_MOE_HG_KEY_MAP.pop(_dense_key, None)


def convert_qwen35_moe_state_dict(
    state_dict: dict[str, object], config: Qwen35MoeConfig
) -> dict[str, object]:
    if "model.embed_tokens.weight" in state_dict:
        state_dict = convert_state_dict(state_dict, _QWEN35_MOE_HG_KEY_MAP)

    # Convert (1+w) RMSNorm weights to standard (w) by adding 1.0.
    for key in list(state_dict.keys()):
        if any(key.endswith(suffix) for suffix in _QWEN35_RMSNORM_KEYS):
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor):
                state_dict[key] = weight + 1.0

    if config.tied_embeddings:
        if "decoder_frontend.embed.weight" in state_dict:
            state_dict["final_proj.weight"] = state_dict[
                "decoder_frontend.embed.weight"
            ]
        elif "final_proj.weight" in state_dict:
            state_dict["decoder_frontend.embed.weight"] = state_dict[
                "final_proj.weight"
            ]

    return state_dict

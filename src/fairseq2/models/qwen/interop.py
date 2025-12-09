# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, final

from typing_extensions import override

from fairseq2.models.utils.hg import HuggingFaceConfig, HuggingFaceConverter
from fairseq2.models.qwen.config import QwenConfig
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

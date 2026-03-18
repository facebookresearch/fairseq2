# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, final

from typing_extensions import override

from fairseq2.models.hg import HuggingFaceConfig, HuggingFaceConverter
from fairseq2.models.olmo.config import OLMOConfig
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.utils.config import cast_config_type

_HG_KEY_MAP: Final = {
    # fmt: off
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":
        r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":
        r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":
        r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":
        r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":
        r"decoder.layers.\1.self_attn.q_norm.",  # OLMO Q/K Norm
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":
        r"decoder.layers.\1.self_attn.k_norm.",  # OLMO Q/K Norm
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.":
        r"decoder.layers.\1.self_attn_layer_norm.",  # Post-Norm
    r"^model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":
        r"decoder.layers.\1.ffn_layer_norm.",  # Post-Norm after FFN
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":
        r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":
        r"decoder.layers.\1.ffn.output_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":
        r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.norm\.":
        r"decoder.layer_norm.",
    r"^model\.embed_tokens\.":
        r"decoder_frontend.embed.",
    r"^lm_head\.":
        r"final_proj.",
    # fmt: on
}


def convert_olmo_state_dict(
    state_dict: dict[str, object], config: OLMOConfig
) -> dict[str, object]:
    """Convert OLMO state dictionary from HuggingFace format to fairseq2 format.

    OLMO uses Post-Norm architecture, so:
    - No input_layernorm (Pre-Norm)
    - Has post_attention_layernorm (after attention)
    - Has post_feedforward_layernorm (after FFN)
    - Has Q/K Norm in attention layers

    .. note::
        Only the HuggingFace format is supported. For OLMO's original checkpoint
        format, convert to HuggingFace format first using the official OLMO
        conversion tools before loading.
    """
    if "model.embed_tokens.weight" in state_dict:  # HuggingFace format
        state_dict = convert_state_dict(state_dict, _HG_KEY_MAP)

    # Handle tied embeddings
    if config.tied_embeddings:
        state_dict["final_proj.weight"] = state_dict["decoder_frontend.embed.weight"]

    return state_dict


@final
class _OLMOHuggingFaceConverter(HuggingFaceConverter):
    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, OLMOConfig)

        is_olmo3 = (
            config.sliding_window is not None or config.yarn_scale_config is not None
        )

        data: dict[str, object] = {
            "hidden_size": config.model_dim,
            "max_position_embeddings": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tied_embeddings,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_attn_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.ffn_inner_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "pad_token_id": config.pad_idx,
            "eos_token_id": config.eos_token_id,
            "bos_token_id": config.bos_token_id,
        }

        if is_olmo3:
            rope_scaling: dict[str, object] = {
                "rope_theta": config.rope_theta,
                "rope_type": "default",
            }

            if config.yarn_scale_config is not None:
                yarn = config.yarn_scale_config
                rope_scaling = {
                    "rope_type": "yarn",
                    "rope_theta": config.rope_theta,
                    "factor": yarn.scale_factor,
                    "original_max_position_embeddings": yarn.original_max_seq_len,
                    "beta_fast": yarn.beta_fast,
                    "beta_slow": yarn.beta_slow,
                    "mscale": yarn.mscale,
                    "mscale_all_dim": yarn.mscale_all_dim,
                }

            data["rope_scaling"] = rope_scaling
            data["sliding_window"] = config.sliding_window

            if config.layer_types is not None:
                data["layer_types"] = config.layer_types

            return HuggingFaceConfig(
                data, kls_name="Olmo3Config", arch="Olmo3ForCausalLM"
            )
        else:
            data["rope_scaling"] = {
                "rope_theta": config.rope_theta,
                "rope_type": "default",
            }

            return HuggingFaceConfig(
                data, kls_name="Olmo2Config", arch="Olmo2ForCausalLM"
            )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, OLMOConfig)

        key_map = create_reverse_key_map(_HG_KEY_MAP)

        hg_state_dict = convert_state_dict(state_dict, key_map)

        if config.tied_embeddings:
            del hg_state_dict["lm_head.weight"]

        return hg_state_dict

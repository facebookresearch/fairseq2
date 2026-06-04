# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict conversion between HuggingFace NemotronH and fairseq2 format.

HuggingFace key structure:
    language_model.backbone.embeddings.weight
    language_model.backbone.layers.{i}.norm.weight
    language_model.backbone.layers.{i}.mixer.*  (type-dependent)
    language_model.backbone.norm_f.weight
    language_model.lm_head.weight

fairseq2 key structure:
    decoder_frontend.embed.weight
    decoder.layers.{i}.norm.weight
    decoder.layers.{i}.mixer.*  (preserved as-is)
    decoder.layer_norm.weight
    final_proj.weight

Note: For Mamba2 layers, the mixer keys map directly:
    mixer.in_proj.weight -> mixer.in_proj.weight
    mixer.conv1d.weight  -> mixer.conv1d.weight
    mixer.A_log          -> mixer.A_log
    mixer.D              -> mixer.D
    mixer.dt_bias        -> mixer.dt_bias
    mixer.norm.weight    -> mixer.norm.weight
    mixer.out_proj.weight -> mixer.out_proj.weight

For Attention layers:
    mixer.q_proj.weight  -> mixer.q_proj.weight
    mixer.k_proj.weight  -> mixer.k_proj.weight
    mixer.v_proj.weight  -> mixer.v_proj.weight
    mixer.o_proj.weight  -> mixer.output_proj.weight

For MoE layers:
    mixer.gate.weight                      -> mixer.gate.weight
    mixer.gate.e_score_correction_bias     -> mixer.gate.e_score_correction_bias
    mixer.experts.{j}.up_proj.weight       -> mixer.experts.{j}.up_proj.weight
    mixer.experts.{j}.down_proj.weight     -> mixer.experts.{j}.down_proj.weight
    mixer.shared_experts.up_proj.weight    -> mixer.shared_experts.up_proj.weight
    mixer.shared_experts.down_proj.weight  -> mixer.shared_experts.down_proj.weight
"""

from __future__ import annotations

from typing import Final, final

from typing_extensions import override

from fairseq2.models.hg import HuggingFaceConfig, HuggingFaceConverter
from fairseq2.models.nemotron.config import NemotronHConfig
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.utils.config import cast_config_type

# Key mapping from HuggingFace to fairseq2
# Only the prefix and a few specific keys need remapping;
# the mixer subkeys are mostly preserved.
_HG_KEY_MAP: Final = {
    # fmt: off
    # Embeddings
    r"^language_model\.backbone\.embeddings\.":                                      r"decoder_frontend.embed.",
    # Final norm
    r"^language_model\.backbone\.norm_f\.":                                          r"decoder.layer_norm.",
    # LM head
    r"^language_model\.lm_head\.":                                                   r"final_proj.",
    # Per-layer norm (pre-norm applied before mixer)
    r"^language_model\.backbone\.layers\.([0-9]+)\.norm\.":                           r"decoder.layers.\1.norm.",
    # Attention layers: o_proj -> output_proj
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.o_proj\.":                  r"decoder.layers.\1.mixer.output_proj.",
    # Attention layers: q/k/v proj (no remapping needed on name, just prefix)
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.q_proj\.":                  r"decoder.layers.\1.mixer.q_proj.",
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.k_proj\.":                  r"decoder.layers.\1.mixer.k_proj.",
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.v_proj\.":                  r"decoder.layers.\1.mixer.v_proj.",
    # MoE layers: gate (router)
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.gate\.":                    r"decoder.layers.\1.mixer.gate.",
    # MoE layers: experts
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.experts\.":                 r"decoder.layers.\1.mixer.experts.",
    # MoE layers: shared experts
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.shared_experts\.":          r"decoder.layers.\1.mixer.shared_experts.",
    # Mamba2 layers: all mixer subkeys (in_proj, conv1d, A_log, D, dt_bias, norm, out_proj)
    r"^language_model\.backbone\.layers\.([0-9]+)\.mixer\.":                          r"decoder.layers.\1.mixer.",
    # fmt: on
}

# Keys to skip during conversion (multimodal components for Phase 1)
_SKIP_PREFIXES: Final = [
    "vision_model.",
    "mlp1.",
    "sound_encoder.",
    "sound_projection.",
]


def convert_nemotron_h_state_dict(
    state_dict: dict[str, object], config: NemotronHConfig
) -> dict[str, object]:
    """Convert a NemotronH state dict to fairseq2 format.

    Handles both HuggingFace format (with language_model.backbone prefix)
    and already-converted fairseq2 format.

    Args:
        state_dict: The state dict to convert.
        config: The NemotronH configuration.

    Returns:
        The converted state dict.
    """
    # Check if this is HuggingFace format
    if any(k.startswith("language_model.") for k in state_dict.keys()):
        # Skip multimodal keys for now (Phase 1 = text only)
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
                continue
            filtered_state_dict[key] = value

        state_dict = convert_state_dict(filtered_state_dict, _HG_KEY_MAP)

    return state_dict


@final
class _NemotronHHuggingFaceConverter(HuggingFaceConverter):
    """Converts between fairseq2 and HuggingFace state dict formats."""

    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, NemotronHConfig)

        data: dict[str, object] = {
            "hidden_size": config.model_dim,
            "max_position_embeddings": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tied_embeddings,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_attn_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "rms_norm_eps": config.rms_norm_eps,
            "n_routed_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "mamba_num_heads": config.mamba_num_heads,
            "mamba_head_dim": config.mamba_head_dim,
            "ssm_state_size": config.ssm_state_size,
        }

        return HuggingFaceConfig(
            data,
            kls_name="NemotronHConfig",
            arch="NemotronHForCausalLM",
        )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, NemotronHConfig)

        key_map = create_reverse_key_map(_HG_KEY_MAP)
        hg_state_dict = convert_state_dict(state_dict, key_map)

        return hg_state_dict

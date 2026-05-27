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

# Audio key mapping (HF sound_encoder/sound_projection -> fairseq2)
# In the multimodal model, audio modules are at the top level:
#   sound_encoder.encoder.* -> sound_encoder.*
#   sound_projection.* -> sound_projection.*
_HG_AUDIO_KEY_MAP: Final = {
    # fmt: off
    # === Sound encoder: strip "encoder." prefix ===
    # Subsampling layers (direct mapping, just strip "encoder.")
    r"^sound_encoder\.encoder\.subsampling\.":                                       r"sound_encoder.subsampling.",

    # Conformer layers: FFN linear1/linear2 -> inner_proj/output_proj
    # (StandardFeedForwardNetwork uses inner_proj/output_proj)
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.feed_forward1\.linear1\.":          r"sound_encoder.layers.\1.feed_forward1.inner_proj.",
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.feed_forward1\.linear2\.":          r"sound_encoder.layers.\1.feed_forward1.output_proj.",
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.feed_forward2\.linear1\.":          r"sound_encoder.layers.\1.feed_forward2.inner_proj.",
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.feed_forward2\.linear2\.":          r"sound_encoder.layers.\1.feed_forward2.output_proj.",

    # Conformer layers: self_attn o_proj -> output_proj
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.self_attn\.o_proj\.":               r"sound_encoder.layers.\1.self_attn.output_proj.",

    # Conformer layers: conv norm -> batch_norm (ConformerConvolution naming)
    r"^sound_encoder\.encoder\.layers\.([0-9]+)\.conv\.norm\.":                      r"sound_encoder.layers.\1.conv.batch_norm.",

    # Conformer layers: all other keys (norms, self_attn q/k/v/bias_u/v, conv, etc.)
    # Just strip "encoder." prefix
    r"^sound_encoder\.encoder\.layers\.":                                            r"sound_encoder.layers.",

    # === Sound projection (direct mapping — same names) ===
    r"^sound_projection\.":                                                          r"sound_projection.",
    # fmt: on
}

# Keys to skip during conversion (multimodal components not yet implemented)
_SKIP_PREFIXES_TEXT_ONLY: Final = [
    "vision_model.",
    "mlp1.",
    "sound_encoder.",
    "sound_projection.",
]

# Keys to skip when audio is enabled (still skip vision)
_SKIP_PREFIXES_AUDIO: Final = [
    "vision_model.",
    "mlp1.",
    # Non-persistent buffers that are not in the state dict
    "sound_encoder.encoder.feature_extractor.",
    "sound_encoder.encoder.encode_positions.",
]


def convert_nemotron_h_state_dict(
    state_dict: dict[str, object], config: NemotronHConfig
) -> dict[str, object]:
    """Convert a NemotronH state dict to fairseq2 format.

    Handles both HuggingFace format (with language_model.backbone prefix)
    and already-converted fairseq2 format.

    When ``config.audio_config`` is set, audio keys (sound_encoder.*,
    sound_projection.*) are converted instead of skipped.

    Args:
        state_dict: The state dict to convert.
        config: The NemotronH configuration.

    Returns:
        The converted state dict.
    """
    # Check if this is HuggingFace format
    if any(k.startswith("language_model.") for k in state_dict.keys()):
        has_audio = config.audio_config is not None
        skip_prefixes = _SKIP_PREFIXES_AUDIO if has_audio else _SKIP_PREFIXES_TEXT_ONLY

        # Filter out keys we don't handle
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                continue
            filtered_state_dict[key] = value

        # Convert LM keys
        state_dict = convert_state_dict(filtered_state_dict, _HG_KEY_MAP)

        # If audio is enabled, convert audio keys separately
        # (they don't have the language_model prefix, so they pass through
        # the LM key map unchanged — apply audio map to those)
        if has_audio:
            state_dict = convert_state_dict(state_dict, _HG_AUDIO_KEY_MAP)

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

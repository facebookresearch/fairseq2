# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final

import torch

from fairseq2.models.family import HuggingFaceExport
from fairseq2.models.gemma3n.config import Gemma3nConfig
from fairseq2.models.utils.checkpoint import (
    convert_state_dict,
    create_reverse_key_map,
)

# HuggingFace → fairseq2 key mappings
_HG_KEY_MAP: Final = {
    # fmt: off
    # Embedding layers
    r"^model\.language_model\.embed_tokens\.":                        "decoder_frontend.embed.",
    r"^lm_head\.":                                                     "final_proj.proj.",

    # Audio tower - subsample convolution projection
    r"^model\.audio_tower\.subsample_conv_projection\.conv_0\.conv\.": "audio_tower.subsample.conv_0.",
    r"^model\.audio_tower\.subsample_conv_projection\.conv_0\.norm\.": "audio_tower.subsample.norm_0.",
    r"^model\.audio_tower\.subsample_conv_projection\.conv_1\.conv\.": "audio_tower.subsample.conv_1.",
    r"^model\.audio_tower\.subsample_conv_projection\.conv_1\.norm\.": "audio_tower.subsample.norm_1.",
    r"^model\.audio_tower\.subsample_conv_projection\.input_proj_linear\.": "audio_tower.subsample.proj.",

    # Audio tower - conformer layers (attention)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.pre_attn_norm\.": r"audio_tower.encoder.layers.\1.self_attn_layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.q_proj\.": r"audio_tower.encoder.layers.\1.self_attn.q_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.k_proj\.": r"audio_tower.encoder.layers.\1.self_attn.k_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.v_proj\.": r"audio_tower.encoder.layers.\1.self_attn.v_proj.",
    # per_dim_scale: fs2 uses learned per-dim scaling; HF does not.
    # Kept in key map for loading; filtered out during export in _convert_to_hg_state_dict.
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.per_dim_scale$": r"audio_tower.encoder.layers.\1.self_attn.sdpa.per_dim_scale",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.relative_position_embedding\.pos_proj\.": r"audio_tower.encoder.layers.\1.self_attn.sdpa.pos_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.post\.": r"audio_tower.encoder.layers.\1.self_attn.output_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.post_norm\.": r"audio_tower.encoder.layers.\1.self_attn_post_norm.",

    # Audio tower - conformer layers (FFN start)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.pre_layer_norm\.": r"audio_tower.encoder.layers.\1.ffn1_layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.ffw_layer_1\.": r"audio_tower.encoder.layers.\1.ffn1.inner_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.ffw_layer_2\.": r"audio_tower.encoder.layers.\1.ffn1.output_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.post_layer_norm\.": r"audio_tower.encoder.layers.\1.ffn1_post_layer_norm.",

    # Audio tower - conformer layers (convolution)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.lconv1d\.pre_layer_norm\.": r"audio_tower.encoder.layers.\1.conv_layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.lconv1d\.linear_start\.": r"audio_tower.encoder.layers.\1.conv.pointwise_conv1.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.lconv1d\.depthwise_conv1d\.": r"audio_tower.encoder.layers.\1.conv.depthwise_conv.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.lconv1d\.conv_norm\.": r"audio_tower.encoder.layers.\1.conv.layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.lconv1d\.linear_end\.": r"audio_tower.encoder.layers.\1.conv.pointwise_conv2.",

    # Audio tower - conformer layers (FFN end)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_end\.pre_layer_norm\.": r"audio_tower.encoder.layers.\1.ffn2_layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_end\.ffw_layer_1\.": r"audio_tower.encoder.layers.\1.ffn2.inner_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_end\.ffw_layer_2\.": r"audio_tower.encoder.layers.\1.ffn2.output_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_end\.post_layer_norm\.": r"audio_tower.encoder.layers.\1.ffn2_post_layer_norm.",

    # Audio tower - conformer layers (final norm)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.norm\.": r"audio_tower.encoder.layers.\1.layer_norm.",

    # Multimodal embedder (audio → text projection)
    r"^model\.embed_audio\.embedding\.": "audio_tower.embedder.embedding.",
    r"^model\.embed_audio\.hard_embedding_norm\.": "audio_tower.embedder.hard_embedding_norm.",
    r"^model\.embed_audio\.soft_embedding_norm\.": "audio_tower.embedder.soft_embedding_norm.",
    r"^model\.embed_audio\.embedding_projection\.": "audio_tower.embedder.embedding_projection.",
    r"^model\.embed_audio\.embedding_post_projection_norm\.": "audio_tower.embedder.embedding_post_projection_norm.",

    # Decoder layers - attention with QK norm
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.": r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.": r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.": r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.": r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.": r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.": r"decoder.layers.\1.self_attn.k_norm.",

    # Decoder layers - normalization
    r"^model\.language_model\.layers\.([0-9]+)\.input_layernorm\.":              r"decoder.layers.\1.input_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.":     r"decoder.layers.\1.post_attention_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.":    r"decoder.layers.\1.pre_feedforward_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":   r"decoder.layers.\1.post_feedforward_layernorm.",

    # Decoder layers - LAuReL
    r"^model\.language_model\.layers\.([0-9]+)\.laurel\.linear_left\.":          r"decoder.layers.\1.laurel.linear_left.",
    r"^model\.language_model\.layers\.([0-9]+)\.laurel\.linear_right\.":         r"decoder.layers.\1.laurel.linear_right.",
    r"^model\.language_model\.layers\.([0-9]+)\.laurel\.post_laurel_norm\.":     r"decoder.layers.\1.laurel.post_laurel_norm.",

    # Decoder layers - AltUp (per-layer predict/correct)
    # correct_output_scale: fs2 stores as param; HF derives from config.
    # Kept in key map for loading; filtered out during export in _convert_to_hg_state_dict.
    r"^model\.language_model\.layers\.([0-9]+)\.altup\.correct_output_scale$":   r"decoder.layers.\1.altup.correct_output_scale",
    r"^model\.language_model\.layers\.([0-9]+)\.altup\.correction_coefs\.":      r"decoder.layers.\1.altup.correction_coefs.",
    r"^model\.language_model\.layers\.([0-9]+)\.altup\.prediction_coefs\.":      r"decoder.layers.\1.altup.prediction_coefs.",
    r"^model\.language_model\.layers\.([0-9]+)\.altup\.modality_router\.":       r"decoder.layers.\1.altup.modality_router.",
    r"^model\.language_model\.layers\.([0-9]+)\.altup\.router_norm\.":           r"decoder.layers.\1.altup.router_norm.",

    # Decoder layers - PLE (per-layer gating and projection)
    r"^model\.language_model\.layers\.([0-9]+)\.per_layer_input_gate\.":         r"decoder.layers.\1.per_layer_input_gate.",
    r"^model\.language_model\.layers\.([0-9]+)\.per_layer_projection\.":         r"decoder.layers.\1.per_layer_projection.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_per_layer_input_norm\.":    r"decoder.layers.\1.post_per_layer_input_norm.",

    # Decoder layers - FFN (standard and AltUp)
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.":   r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.":     r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.":   r"decoder.layers.\1.ffn.output_proj.",

    # Model-level: AltUp projections
    r"^model\.language_model\.altup_projections\.([0-9]+)\.":         r"decoder.altup_projections.\1.",
    r"^model\.language_model\.altup_unembed_projections\.([0-9]+)\.": r"decoder.altup_unembed_projections.\1.",

    # Model-level: PLE embeddings
    r"^model\.language_model\.embed_tokens_per_layer\.":              "decoder_frontend.embed_tokens_per_layer.",
    r"^model\.language_model\.per_layer_model_projection\.":          "decoder_frontend.per_layer_model_projection.",
    r"^model\.language_model\.per_layer_projection_norm\.":           "decoder_frontend.per_layer_projection_norm.",

    # Final normalization
    r"^model\.language_model\.norm\.":                                "decoder.layer_norm.",
    # fmt: on
}


def convert_gemma3n_state_dict(
    state_dict: dict[str, object],
    config: Gemma3nConfig,
) -> dict[str, object]:
    """Convert a HuggingFace Gemma3n state dictionary to fairseq2 format.

    :param state_dict: The HuggingFace Gemma3n state dictionary.
    :param config: The Gemma3n configuration.
    :returns: The fairseq2-compatible state dictionary.

    Filters out vision tower (not yet integrated). Audio tower is included
    when config.audio_config is set.
    """
    # Filter out multimodal components not yet integrated
    vision_prefixes = (
        "model.vision_tower.",
        "model.embed_vision.",
    )
    audio_prefixes = (
        "model.audio_tower.",
        "model.embed_audio.",
    )
    # HF's embedding_post_projection_norm uses with_scale=False (non-learnable
    # scalar buffer), so there's no weight to load.
    audio_buffer_prefixes = ("model.embed_audio.embedding_post_projection_norm.",)

    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(vision_prefixes)
        and not k.startswith(audio_buffer_prefixes)
        and (config.audio_config is not None or not k.startswith(audio_prefixes))
    }

    converted = convert_state_dict(filtered_state_dict, _HG_KEY_MAP)

    # Reshape pointwise conv weights: HF Linear [N, M] → fs2 Conv1d [N, M, 1]
    for key, value in converted.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and ("pointwise_conv1.weight" in key or "pointwise_conv2.weight" in key)
        ):
            converted[key] = value.unsqueeze(-1)

    return converted


def export_gemma3n(
    state_dict: dict[str, object], config: Gemma3nConfig
) -> HuggingFaceExport:
    """Export a fairseq2 Gemma3n model to HuggingFace format.

    :param state_dict: The fairseq2 state dictionary.
    :param config: The Gemma3n configuration.
    :returns: A :class:`HuggingFaceExport` with HF state dict and config.
    """
    hg_state_dict = _convert_to_hg_state_dict(state_dict)
    hg_config = _convert_to_hg_config(config)

    return HuggingFaceExport(
        hg_state_dict,
        hg_config,
        config_kls_name="Gemma3nConfig",
        arch="Gemma3nForConditionalGeneration",
    )


def _convert_to_hg_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    state_dict = dict(state_dict)

    # Remove keys that exist in fs2 but not in HF's model
    _FS2_ONLY_SUFFIXES = (
        # fs2 learned per-dim scaling; HF uses 1/sqrt(head_dim)
        ".per_dim_scale",
        # fs2 stores as param; HF derives from altup_correct_scale config
        ".correct_output_scale",
    )
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(k.endswith(s) for s in _FS2_ONLY_SUFFIXES)
    }

    # Squeeze pointwise conv weights: fs2 Conv1d [N, M, 1] → HF Linear [N, M]
    for key, value in state_dict.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 3
            and ("pointwise_conv1.weight" in key or "pointwise_conv2.weight" in key)
        ):
            state_dict[key] = value.squeeze(-1)

    key_map = create_reverse_key_map(_HG_KEY_MAP)

    return convert_state_dict(state_dict, key_map)


def _convert_to_hg_config(config: Gemma3nConfig) -> dict[str, object]:
    from transformers import Gemma3nTextConfig

    text_config = Gemma3nTextConfig(  # type: ignore[call-arg]
        hidden_size=config.model_dim,
        intermediate_size=config.ffn_inner_dim,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attn_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=config.rms_norm_eps,
        sliding_window=config.sliding_window,
        final_logit_softcapping=config.final_logit_soft_cap,
        tie_word_embeddings=config.tied_embeddings,
        num_kv_shared_layers=config.num_kv_shared_layers,
        laurel_rank=config.laurel_rank,
        altup_num_inputs=config.altup_num_inputs,
        altup_active_idx=config.altup_active_idx,
        altup_coef_clip=config.altup_coef_clip,
        altup_correct_scale=config.altup_correct_scale,
        vocab_size_per_layer_input=config.vocab_size_per_layer_input,
        hidden_size_per_layer_input=config.hidden_size_per_layer_input,
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": config.rope_theta,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": config.rope_theta_global,
            },
        },
    )

    return {
        "model_type": "gemma3n",
        "text_config": text_config,
    }

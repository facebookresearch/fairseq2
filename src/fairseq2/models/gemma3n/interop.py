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
    r"^model\.language_model\.embed_tokens\.":                        "decoder_frontend.embed.",
    r"^lm_head\.":                                                     "final_proj.",

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
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.per_dim_scale$": r"audio_tower.encoder.layers.\1.self_attn.sdpa.per_dim_scale",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.attn\.relative_position_embedding\.pos_proj\.": r"audio_tower.encoder.layers.\1.self_attn.sdpa.rel_k_embed.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.attention\.post\.": r"audio_tower.encoder.layers.\1.self_attn.output_proj.",

    # Audio tower - conformer layers (FFN start)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.pre_layer_norm\.": r"audio_tower.encoder.layers.\1.ffn1_layer_norm.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.ffw_layer_1\.": r"audio_tower.encoder.layers.\1.ffn1.inner_proj.",
    r"^model\.audio_tower\.conformer\.([0-9]+)\.ffw_layer_start\.ffw_layer_2\.": r"audio_tower.encoder.layers.\1.ffn1.output_proj.",

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

    # Audio tower - conformer layers (final norm)
    r"^model\.audio_tower\.conformer\.([0-9]+)\.norm\.": r"audio_tower.encoder.layers.\1.layer_norm.",

    # Multimodal embedder (audio → text projection)
    r"^model\.embed_audio\.embedding\.": "embed_audio.embedding.",
    r"^model\.embed_audio\.hard_embedding_norm\.": "embed_audio.hard_embedding_norm.",
    r"^model\.embed_audio\.soft_embedding_norm\.": "embed_audio.soft_embedding_norm.",
    r"^model\.embed_audio\.embedding_projection\.": "embed_audio.embedding_projection.",
    r"^model\.embed_audio\.embedding_post_projection_norm\.": "embed_audio.embedding_post_projection_norm.",

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

    Args:
        state_dict: The HuggingFace Gemma3n state dictionary.
        config: The Gemma3n configuration.

    Returns:
        The fairseq2-compatible state dictionary.

    Notes:
        Supports audio tower (subsample + conformer encoder) and multimodal embedder.
        Vision tower (embed_vision, vision_tower) is filtered out as not yet implemented.
    """
    import torch

    # Filter out vision components (not implemented yet)
    vision_prefixes = (
        "model.vision_tower.",
        "model.embed_vision.",
    )

    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(vision_prefixes)
    }

    converted = convert_state_dict(filtered_state_dict, _HG_KEY_MAP)

    # Post-process audio tower weights for architecture differences
    # 1. Reshape Conv1d pointwise weights from (out, in) to (out, in, 1)
    for key in list(converted.keys()):
        if "audio_tower.encoder" in key and ("pointwise_conv1.weight" in key or "pointwise_conv2.weight" in key):
            # HF stores as Linear (out, in), fairseq2 Conv1d expects (out, in, 1)
            converted[key] = converted[key].unsqueeze(-1)

    # 2. Convert Shaw relative position embeddings from linear projection to embedding table
    # HF: linear projection (hidden_size, hidden_size) = (1536, 1536)
    # FS2: embedding table (num_pos, head_dim) = (14, 192)
    # The HF linear is actually just a transposed embedding lookup
    for key in list(converted.keys()):
        if "audio_tower.encoder" in key and "sdpa.rel_k_embed.weight" in key:
            # Extract (num_pos, head_dim) from (hidden_size, hidden_size)
            # The HF weight is (1536, 1536) but we only need (14, 192)
            # where 14 = max_left + 1 + max_right = 13 + 1 + 0
            # and 192 = head_dim = 1536 / 8
            linear_weight = converted[key]  # (1536, 1536)
            num_heads = 8
            head_dim = 1536 // num_heads  # 192
            max_left = 13
            max_right = 0
            num_pos = max_left + 1 + max_right  # 14

            # The linear weight is transposed - transpose it and extract the first num_pos x head_dim
            # Actually, HF applies this as: output = input @ weight.T
            # We want embedding lookup which is: output = embedding_table[indices]
            # So we need to extract the first num_pos rows and num_heads * head_dim columns
            # But the weight is stored as (out_features, in_features) = (1536, 1536)
            # For our embedding, we need (num_pos, head_dim)
            # The first num_pos columns of the transposed weight should work
            embedding_weight = linear_weight.T[:num_pos, :head_dim].contiguous()
            converted[key] = embedding_weight

    return converted


def convert_to_hf_gemma3n_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert a fairseq2 Gemma3n state dictionary to HuggingFace format.

    Args:
        state_dict: The fairseq2 Gemma3n state dictionary.

    Returns:
        The HuggingFace-compatible state dictionary.
    """
    raise NotImplementedError(
        "fairseq2 → HuggingFace conversion not yet implemented."
    )

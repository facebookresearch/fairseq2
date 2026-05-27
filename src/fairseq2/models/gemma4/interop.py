# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace ↔ fairseq2 state-dict conversion for Gemma 4."""

from __future__ import annotations

import re
from typing import Final, final

from typing_extensions import override

from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.models.gemma4.config import Gemma4Config, get_kv_projection_role
from fairseq2.models.hg import HuggingFaceConfig, HuggingFaceConverter
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.utils.config import cast_config_type

# HuggingFace → fairseq2 key mappings.
#
# The regex keys match HF parameter names and the string values are the
# corresponding fairseq2 names.  Numeric layer indices are captured via
# ``([0-9]+)`` groups.
_HG_KEY_MAP: Final = {
    # fmt: off
    # ---- Embedding ----
    r"^model\.language_model\.embed_tokens\.":                          "decoder_frontend.embed.",
    r"^lm_head\.":                                                      "final_proj.proj.",

    # ---- Decoder layers — self-attention ----
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.":   r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.":   r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.":   r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.":   r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.":   r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.":   r"decoder.layers.\1.self_attn.k_norm.",
    r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.v_norm\.":   r"decoder.layers.\1.self_attn.v_norm.",

    # ---- Decoder layers — layer norms ----
    r"^model\.language_model\.layers\.([0-9]+)\.input_layernorm\.":              r"decoder.layers.\1.input_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.":     r"decoder.layers.\1.post_attention_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.":    r"decoder.layers.\1.pre_feedforward_layernorm.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":   r"decoder.layers.\1.post_feedforward_layernorm.",

    # ---- Decoder layers — FFN (dense MLP) ----
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.":   r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.":     r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.":   r"decoder.layers.\1.ffn.output_proj.",

    # ---- Decoder layers — PLE (Per-Layer Embeddings) ----
    r"^model\.language_model\.layers\.([0-9]+)\.per_layer_input_gate\.":      r"decoder.layers.\1.per_layer_input_gate.",
    r"^model\.language_model\.layers\.([0-9]+)\.per_layer_projection\.":      r"decoder.layers.\1.per_layer_projection.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_per_layer_input_norm\.": r"decoder.layers.\1.post_per_layer_input_norm.",

    # ---- Decoder layers — layer scalar ----
    # Note: no trailing `$` anchor — ``create_reverse_key_map`` would put a
    # literal ``$`` in the reversed replacement string.  The patterns are
    # specific enough that no false matches occur.
    r"^model\.language_model\.layers\.([0-9]+)\.layer_scalar":              r"decoder.layers.\1.layer_scalar",

    # ---- Decoder layers — MoE router ----
    r"^model\.language_model\.layers\.([0-9]+)\.router\.norm\.":              r"decoder.layers.\1.router.norm.",
    r"^model\.language_model\.layers\.([0-9]+)\.router\.proj\.":              r"decoder.layers.\1.router.proj.",
    r"^model\.language_model\.layers\.([0-9]+)\.router\.scale":              r"decoder.layers.\1.router.scale",
    r"^model\.language_model\.layers\.([0-9]+)\.router\.per_expert_scale":   r"decoder.layers.\1.router.per_expert_scale",

    # ---- Decoder layers — MoE experts ----
    # gate_up_proj and down_proj are raw Parameters (not nn.Linear), so no
    # ".weight" suffix in the HF keys.
    r"^model\.language_model\.layers\.([0-9]+)\.experts\.gate_up_proj":      r"decoder.layers.\1.experts.gate_up_proj",
    r"^model\.language_model\.layers\.([0-9]+)\.experts\.down_proj":         r"decoder.layers.\1.experts.down_proj",

    # ---- Decoder layers — MoE extra norms ----
    r"^model\.language_model\.layers\.([0-9]+)\.post_feedforward_layernorm_1\.":   r"decoder.layers.\1.post_feedforward_layernorm_1.",
    r"^model\.language_model\.layers\.([0-9]+)\.pre_feedforward_layernorm_2\.":    r"decoder.layers.\1.pre_feedforward_layernorm_2.",
    r"^model\.language_model\.layers\.([0-9]+)\.post_feedforward_layernorm_2\.":   r"decoder.layers.\1.post_feedforward_layernorm_2.",

    # ---- Model-level — PLE embeddings ----
    r"^model\.language_model\.embed_tokens_per_layer\.":              "decoder_frontend.embed_tokens_per_layer.",
    r"^model\.language_model\.per_layer_model_projection\.":          "decoder_frontend.per_layer_model_projection.",
    r"^model\.language_model\.per_layer_projection_norm\.":           "decoder_frontend.per_layer_projection_norm.",

    # ---- Final normalization ----
    r"^model\.language_model\.norm\.":                                "decoder.layer_norm.",

    # =========================================================================
    # Audio tower — subsample conv projection
    # =========================================================================
    r"^model\.audio_tower\.subsample_conv_projection\.layer0\.conv\.":      "audio_tower.subsample.conv_0.",
    r"^model\.audio_tower\.subsample_conv_projection\.layer0\.norm\.":      "audio_tower.subsample.norm_0.",
    r"^model\.audio_tower\.subsample_conv_projection\.layer1\.conv\.":      "audio_tower.subsample.conv_1.",
    r"^model\.audio_tower\.subsample_conv_projection\.layer1\.norm\.":      "audio_tower.subsample.norm_1.",
    r"^model\.audio_tower\.subsample_conv_projection\.input_proj_linear\.": "audio_tower.subsample.proj.",

    # =========================================================================
    # Audio tower — conformer layers: self-attention
    # Note: HF wraps Linear ops in ClippableLinear, which adds a `.linear.`
    # sub-module for weights.  Clipping buffers (input_min, etc.) are at
    # the ClippableLinear level (no `.linear.`).  Weight rules (with `.linear.`)
    # MUST come before buffer rules (without `.linear.`) since
    # convert_state_dict applies the FIRST matching pattern.
    # =========================================================================
    # Weights (strip .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.q_proj\.linear\.":   r"audio_tower.encoder.layers.\1.self_attn.q_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.k_proj\.linear\.":   r"audio_tower.encoder.layers.\1.self_attn.k_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.v_proj\.linear\.":   r"audio_tower.encoder.layers.\1.self_attn.v_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.post\.linear\.":     r"audio_tower.encoder.layers.\1.self_attn.output_proj.",
    # Non-ClippableLinear attention params
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.per_dim_scale":      r"audio_tower.encoder.layers.\1.self_attn.sdpa.per_dim_scale",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.relative_k_proj\.":  r"audio_tower.encoder.layers.\1.self_attn.sdpa.pos_proj.",
    # Clipping buffers (no .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.q_proj\.":   r"audio_tower.encoder.layers.\1.self_attn.q_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.k_proj\.":   r"audio_tower.encoder.layers.\1.self_attn.k_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.v_proj\.":   r"audio_tower.encoder.layers.\1.self_attn.v_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.self_attn\.post\.":     r"audio_tower.encoder.layers.\1.self_attn.output_proj.",

    # =========================================================================
    # Audio tower — conformer layers: norms
    # =========================================================================
    r"^model\.audio_tower\.layers\.([0-9]+)\.norm_pre_attn\.":     r"audio_tower.encoder.layers.\1.self_attn_layer_norm.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.norm_post_attn\.":    r"audio_tower.encoder.layers.\1.self_attn_post_norm.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.norm_out\.":          r"audio_tower.encoder.layers.\1.layer_norm.",

    # =========================================================================
    # Audio tower — conformer layers: FFN1 / FFN2
    # ffw_layer_1 = inner_proj (up), ffw_layer_2 = output_proj (down)
    # Weight rules (with .linear.) MUST come before buffer rules.
    # =========================================================================
    # FFN1 weights (strip .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.ffw_layer_1\.linear\.":   r"audio_tower.encoder.layers.\1.ffn1.inner_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.ffw_layer_2\.linear\.":   r"audio_tower.encoder.layers.\1.ffn1.output_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.pre_layer_norm\.":        r"audio_tower.encoder.layers.\1.ffn1_layer_norm.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.post_layer_norm\.":       r"audio_tower.encoder.layers.\1.ffn1_post_layer_norm.",
    # FFN1 clipping buffers (no .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.ffw_layer_1\.":   r"audio_tower.encoder.layers.\1.ffn1.inner_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward1\.ffw_layer_2\.":   r"audio_tower.encoder.layers.\1.ffn1.output_proj.",

    # FFN2 weights (strip .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.ffw_layer_1\.linear\.":   r"audio_tower.encoder.layers.\1.ffn2.inner_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.ffw_layer_2\.linear\.":   r"audio_tower.encoder.layers.\1.ffn2.output_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.pre_layer_norm\.":        r"audio_tower.encoder.layers.\1.ffn2_layer_norm.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.post_layer_norm\.":       r"audio_tower.encoder.layers.\1.ffn2_post_layer_norm.",
    # FFN2 clipping buffers (no .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.ffw_layer_1\.":   r"audio_tower.encoder.layers.\1.ffn2.inner_proj.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.feed_forward2\.ffw_layer_2\.":   r"audio_tower.encoder.layers.\1.ffn2.output_proj.",

    # =========================================================================
    # Audio tower — conformer layers: LightConv1d
    # HF linear_start/linear_end are ClippableLinear wrappers.  fairseq2 now
    # uses Gemma4ClippedLinear (Linear, not Conv1d) for pointwise ops — no
    # weight reshape needed.  Depthwise conv is plain Conv1d.
    # Weight rules (with .linear.) MUST come before buffer rules.
    # =========================================================================
    # Pointwise weights (strip .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.linear_start\.linear\.":   r"audio_tower.encoder.layers.\1.conv.pointwise_conv1.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.linear_end\.linear\.":     r"audio_tower.encoder.layers.\1.conv.pointwise_conv2.",
    # Depthwise conv + norm (no ClippableLinear wrapper)
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.depthwise_conv1d\.":       r"audio_tower.encoder.layers.\1.conv.depthwise_conv.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.conv_norm\.":              r"audio_tower.encoder.layers.\1.conv.layer_norm.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.pre_layer_norm\.":         r"audio_tower.encoder.layers.\1.conv_layer_norm.",
    # Pointwise clipping buffers (no .linear.)
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.linear_start\.":   r"audio_tower.encoder.layers.\1.conv.pointwise_conv1.",
    r"^model\.audio_tower\.layers\.([0-9]+)\.lconv1d\.linear_end\.":     r"audio_tower.encoder.layers.\1.conv.pointwise_conv2.",

    # =========================================================================
    # Audio tower — output projection (nn.Linear with bias)
    # =========================================================================
    r"^model\.audio_tower\.output_proj\.":   "audio_tower.output_proj.",

    # =========================================================================
    # Audio embedder
    # =========================================================================
    r"^model\.embed_audio\.embedding_projection\.":   "audio_embedder.embedding_projection.",
    # fmt: on
}


def convert_gemma4_state_dict(
    state_dict: dict[str, object],
    config: Gemma4Config,
) -> dict[str, object]:
    """Convert a HuggingFace Gemma 4 state dictionary to fairseq2 format.

    :param state_dict: The HuggingFace Gemma 4 state dictionary.
    :param config: The Gemma 4 configuration.
    :returns: The fairseq2-compatible state dictionary.

    When ``audio_config`` is ``None`` (text-only), all audio tower and audio
    embedder parameters are filtered out.  When audio is enabled, the audio
    keys are mapped through ``_HG_KEY_MAP``, stripping the ``.linear.``
    sub-module prefix from ClippableLinear wrappers.  ClippableLinear
    clipping buffers (``input_min``, ``input_max``, ``output_min``,
    ``output_max``) are mapped to the corresponding ``Gemma4ClippedLinear``
    buffers in the fairseq2 model.

    When ``tied_embeddings`` is ``True``, the HF checkpoint omits
    ``lm_head.weight`` (it is tied to the embedding).  The fairseq2 model's
    :class:`TiedProjection` still registers the shared weight as its own
    parameter (``final_proj.proj.weight``), so we copy the embedding weight
    into that slot after conversion.
    """
    # Determine which multimodal prefixes to filter out.
    # Always filter vision; only filter audio when not configured.
    multimodal_prefixes = [
        "model.vision_tower.",
        "model.embed_vision.",
        "model.multi_modal_projector.",
    ]
    if config.audio_config is None:
        multimodal_prefixes.extend(
            [
                "model.audio_tower.",
                "model.embed_audio.",
            ]
        )

    filtered: dict[str, object] = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in multimodal_prefixes):
            continue
        filtered[k] = v

    # Drop lm_head.weight when tied — the embedding weight will be copied
    # into the final_proj slot below.
    if config.tied_embeddings:
        filtered.pop("lm_head.weight", None)

    # Drop k_proj, v_proj, k_norm, and v_norm weights for CONSUMER layers.
    # The HF checkpoint retains these weights for all layers but the HF model
    # does NOT instantiate parameters for consumer layers.  Our fairseq2 model
    # also omits these parameters, so we filter them here to match.
    _consumer_kv_re = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.self_attn\."
        r"(?:k_proj|v_proj|k_norm|v_norm)\."
    )
    if config.num_kv_shared_layers > 0:
        layer_types = config.layer_types
        keys_to_drop: list[str] = []
        for key in filtered:
            m = _consumer_kv_re.match(key)
            if m:
                layer_idx = int(m.group(1))
                layer_type = layer_types[layer_idx]
                kv_role = get_kv_projection_role(
                    layer_idx,
                    layer_type,
                    config.num_layers,
                    config.num_kv_shared_layers,
                    layer_types,
                )
                if kv_role == KVProjectionRole.CONSUMER:
                    keys_to_drop.append(key)
        for key in keys_to_drop:
            del filtered[key]

    converted = convert_state_dict(filtered, _HG_KEY_MAP)

    # Handle tied embeddings: copy embedding weight → final_proj.proj.weight
    # so that ``load_state_dict(strict=True)`` succeeds.  The model module
    # hierarchy is:
    #   final_proj: SoftcappedProjection (when softcapping) or TiedProjection
    #     └── proj: TiedProjection       (when softcapping)
    # In both cases the state-dict key is ``final_proj.proj.weight`` (with
    # softcapping) or ``final_proj.weight`` (without).  We derive the key
    # from the presence of ``final_logit_soft_cap``.
    if config.tied_embeddings:
        embed_key = "decoder_frontend.embed.weight"
        if embed_key in converted:
            if config.final_logit_soft_cap is not None:
                proj_key = "final_proj.proj.weight"
            else:
                proj_key = "final_proj.weight"
            converted[proj_key] = converted[embed_key]

    return converted


# Text-only key map for reverse (fs2 → HF) export.
#
# The forward ``_HG_KEY_MAP`` uses ``model.language_model.*`` prefixes (HF
# multimodal checkpoint format).  For the HuggingFace *text-only*
# ``Gemma4ForCausalLM`` model the keys start with plain ``model.*``.  We define
# a separate text-only map to ensure the exported state dict matches the
# canonical ``transformers`` layout.
_GEMMA4_TEXT_KEY_MAP: Final = {
    # fmt: off
    # ---- Embedding ----
    r"^model\.embed_tokens\.":                          "decoder_frontend.embed.",
    r"^lm_head\.":                                      "final_proj.proj.",

    # ---- Decoder layers — self-attention ----
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":   r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":   r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":   r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":   r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":   r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":   r"decoder.layers.\1.self_attn.k_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_norm\.":   r"decoder.layers.\1.self_attn.v_norm.",

    # ---- Decoder layers — layer norms ----
    r"^model\.layers\.([0-9]+)\.input_layernorm\.":              r"decoder.layers.\1.input_layernorm.",
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.":     r"decoder.layers.\1.post_attention_layernorm.",
    r"^model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.":    r"decoder.layers.\1.pre_feedforward_layernorm.",
    r"^model\.layers\.([0-9]+)\.post_feedforward_layernorm\.":   r"decoder.layers.\1.post_feedforward_layernorm.",

    # ---- Decoder layers — FFN (dense MLP) ----
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":   r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":     r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":   r"decoder.layers.\1.ffn.output_proj.",

    # ---- Decoder layers — PLE (Per-Layer Embeddings) ----
    r"^model\.layers\.([0-9]+)\.per_layer_input_gate\.":      r"decoder.layers.\1.per_layer_input_gate.",
    r"^model\.layers\.([0-9]+)\.per_layer_projection\.":      r"decoder.layers.\1.per_layer_projection.",
    r"^model\.layers\.([0-9]+)\.post_per_layer_input_norm\.": r"decoder.layers.\1.post_per_layer_input_norm.",

    # ---- Decoder layers — layer scalar ----
    r"^model\.layers\.([0-9]+)\.layer_scalar":              r"decoder.layers.\1.layer_scalar",

    # ---- Decoder layers — MoE router ----
    r"^model\.layers\.([0-9]+)\.router\.norm\.":              r"decoder.layers.\1.router.norm.",
    r"^model\.layers\.([0-9]+)\.router\.proj\.":              r"decoder.layers.\1.router.proj.",
    r"^model\.layers\.([0-9]+)\.router\.scale":              r"decoder.layers.\1.router.scale",
    r"^model\.layers\.([0-9]+)\.router\.per_expert_scale":   r"decoder.layers.\1.router.per_expert_scale",

    # ---- Decoder layers — MoE experts ----
    r"^model\.layers\.([0-9]+)\.experts\.gate_up_proj":      r"decoder.layers.\1.experts.gate_up_proj",
    r"^model\.layers\.([0-9]+)\.experts\.down_proj":         r"decoder.layers.\1.experts.down_proj",

    # ---- Decoder layers — MoE extra norms ----
    r"^model\.layers\.([0-9]+)\.post_feedforward_layernorm_1\.":   r"decoder.layers.\1.post_feedforward_layernorm_1.",
    r"^model\.layers\.([0-9]+)\.pre_feedforward_layernorm_2\.":    r"decoder.layers.\1.pre_feedforward_layernorm_2.",
    r"^model\.layers\.([0-9]+)\.post_feedforward_layernorm_2\.":   r"decoder.layers.\1.post_feedforward_layernorm_2.",

    # ---- Model-level — PLE embeddings ----
    r"^model\.embed_tokens_per_layer\.":              "decoder_frontend.embed_tokens_per_layer.",
    r"^model\.per_layer_model_projection\.":          "decoder_frontend.per_layer_model_projection.",
    r"^model\.per_layer_projection_norm\.":           "decoder_frontend.per_layer_projection_norm.",

    # ---- Final normalization ----
    r"^model\.norm\.":                                "decoder.layer_norm.",
    # fmt: on
}


@final
class _Gemma4HuggingFaceConverter(HuggingFaceConverter):
    """Converts fairseq2 Gemma 4 models to HuggingFace Transformers format."""

    @override
    def to_hg_config(self, config: object) -> HuggingFaceConfig:
        config = cast_config_type(config, Gemma4Config)

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
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "sliding_window": config.sliding_window,
            "partial_rotary_factor": config.partial_rotary_factor,
            "hidden_activation": config.hidden_activation,
            "pad_token_id": config.pad_idx,
        }

        # Global attention head parameters.
        data["global_head_dim"] = config.global_head_dim
        data["rope_theta_global"] = config.rope_theta_global
        if config.num_global_key_value_heads is not None:
            data["num_key_value_heads_global"] = config.num_global_key_value_heads

        # K=V and KV sharing.
        data["attention_k_eq_v"] = config.attention_k_eq_v
        data["num_kv_shared_layers"] = config.num_kv_shared_layers
        data["use_double_wide_mlp"] = config.use_double_wide_mlp

        # PLE parameters.
        data["vocab_size_per_layer_input"] = config.vocab_size_per_layer_input
        data["hidden_size_per_layer_input"] = config.hidden_size_per_layer_input

        # Soft-capping.
        if config.final_logit_soft_cap is not None:
            data["final_logit_softcapping"] = config.final_logit_soft_cap

        # Layer types.
        data["layer_types"] = config.layer_types

        # MoE parameters.
        if config.enable_moe:
            data["num_local_experts"] = config.num_experts
            data["num_experts_per_tok"] = config.top_k_experts
            if config.moe_intermediate_size is not None:
                data["moe_intermediate_size"] = config.moe_intermediate_size

        return HuggingFaceConfig(
            data, kls_name="Gemma4TextConfig", arch="Gemma4ForCausalLM"
        )

    @override
    def to_hg_state_dict(
        self, state_dict: dict[str, object], config: object
    ) -> dict[str, object]:
        config = cast_config_type(config, Gemma4Config)

        # Use the text-only key map for export (model.layers.*, not
        # model.language_model.layers.*).
        key_map = create_reverse_key_map(_GEMMA4_TEXT_KEY_MAP)

        hg_state_dict = convert_state_dict(state_dict, key_map)

        # Handle tied embeddings: remove the duplicate lm_head.weight since
        # HF models with tie_word_embeddings=True expect it absent.
        if config.tied_embeddings:
            hg_state_dict.pop("lm_head.weight", None)

        return hg_state_dict

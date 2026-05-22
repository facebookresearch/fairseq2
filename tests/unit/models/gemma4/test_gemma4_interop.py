# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Gemma 4 HuggingFace state-dict interop."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.gemma4.config import (
    Gemma4Config,
    get_gemma4_26b_a4b_config,
    get_gemma4_31b_config,
    get_gemma4_e4b_config,
)
from fairseq2.models.gemma4.factory import create_gemma4_model
from fairseq2.models.gemma4.interop import (
    _GEMMA4_TEXT_KEY_MAP,
    _HG_KEY_MAP,
    _Gemma4HuggingFaceConverter,
    convert_gemma4_state_dict,
)
from fairseq2.models.hg import HuggingFaceConfig
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map


class TestGemma4Interop:
    """Test HF <-> fairseq2 state dict conversion."""

    def _make_small_config(self, *, enable_moe: bool = False) -> Gemma4Config:
        """Create a tiny config for fast testing.

        Uses ``final_logit_soft_cap=30.0`` to match all real Gemma 4 variants.
        With tied embeddings + softcapping, TiedProjection is wrapped inside
        SoftcappedProjection, so no ``final_proj.weight`` key appears in the
        state dict (the weight is shared with the embedding).
        """
        config = Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
            enable_moe=enable_moe,
            num_experts=4 if enable_moe else None,
            top_k_experts=2 if enable_moe else None,
            moe_intermediate_size=16 if enable_moe else None,
        )
        return config

    def _make_small_config_with_ple(self) -> Gemma4Config:
        """Create a tiny config with PLE enabled."""
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=128,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
        )

    def test_state_dict_key_round_trip(self) -> None:
        """fs2 keys -> HF keys -> fs2 keys should be identity."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        # Convert to HF format using reverse key map
        reverse_map = create_reverse_key_map(_HG_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)

        # Verify HF keys have expected prefixes
        for key in hg_state_dict:
            assert key.startswith(
                ("model.", "lm_head.")
            ), f"Unexpected HF key prefix: {key}"

        # Convert back to fs2 format
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _HG_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Round-trip key mismatch.\n"
            f"  Missing in round-trip: {fs2_keys - rt_keys}\n"
            f"  Extra in round-trip:   {rt_keys - fs2_keys}"
        )

    def test_state_dict_key_round_trip_with_moe(self) -> None:
        """MoE model: fs2 keys -> HF keys -> fs2 keys should be identity."""
        config = self._make_small_config(enable_moe=True)

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        reverse_map = create_reverse_key_map(_HG_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _HG_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"MoE round-trip key mismatch.\n"
            f"  Missing: {fs2_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - fs2_keys}"
        )

    def test_state_dict_key_round_trip_with_ple(self) -> None:
        """PLE model: fs2 keys -> HF keys -> fs2 keys should be identity."""
        config = self._make_small_config_with_ple()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        reverse_map = create_reverse_key_map(_HG_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _HG_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"PLE round-trip key mismatch.\n"
            f"  Missing: {fs2_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - fs2_keys}"
        )

    def test_tied_embeddings_no_lm_head(self) -> None:
        """HF checkpoint with tied_embeddings: lm_head.weight is dropped and
        final_proj.proj.weight is populated from the embedding weight."""
        config = self._make_small_config()
        # _make_small_config already has tied_embeddings=True, soft_cap=30.0

        weight = torch.randn(config.vocab_size, config.model_dim)
        hf_state_dict: dict[str, object] = {
            "model.language_model.embed_tokens.weight": weight,
            "model.language_model.norm.weight": torch.zeros(config.model_dim),
            "lm_head.weight": weight.clone(),  # Tied — should be dropped
        }

        result = convert_gemma4_state_dict(dict(hf_state_dict), config)

        # lm_head.weight should have been dropped; instead, the embedding
        # weight is copied into final_proj.proj.weight.
        assert "decoder_frontend.embed.weight" in result
        assert "final_proj.proj.weight" in result
        # Both should be the same tensor (copied from embedding).
        assert (
            result["final_proj.proj.weight"] is result["decoder_frontend.embed.weight"]
        )

    def test_multimodal_keys_filtered(self) -> None:
        """Vision/audio tower keys are filtered out."""
        config = self._make_small_config()
        hf_state_dict: dict[str, object] = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.model_dim
            ),
            "model.language_model.norm.weight": torch.zeros(config.model_dim),
            "model.vision_tower.blocks.0.attn.proj.weight": torch.empty(0),
            "model.audio_tower.encoder.weight": torch.empty(0),
            "model.embed_vision.proj.weight": torch.empty(0),
            "model.embed_audio.proj.weight": torch.empty(0),
            "model.multi_modal_projector.weight": torch.empty(0),
        }

        result = convert_gemma4_state_dict(dict(hf_state_dict), config)

        for key in result:
            assert not key.startswith(
                (
                    "model.vision_tower.",
                    "model.audio_tower.",
                    "model.embed_vision.",
                    "model.embed_audio.",
                    "model.multi_modal_projector.",
                )
            ), f"Multimodal key not filtered: {key}"

    def test_softcapped_projection_key_prefix(self) -> None:
        """When softcapping is enabled, lm_head maps to final_proj.proj.*

        SoftcappedProjection wraps the base projection as ``self.proj``,
        so the state-dict key is ``final_proj.proj.weight`` (not
        ``final_proj.inner.proj.weight``).
        """
        config = self._make_small_config()
        config.final_logit_soft_cap = 30.0
        config.tied_embeddings = False  # So lm_head.weight is present

        hf_state_dict: dict[str, object] = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.model_dim
            ),
            "model.language_model.norm.weight": torch.zeros(config.model_dim),
            "lm_head.weight": torch.randn(config.vocab_size, config.model_dim),
        }

        result = convert_gemma4_state_dict(dict(hf_state_dict), config)

        # lm_head.weight -> final_proj.proj.weight
        assert "final_proj.proj.weight" in result

    def test_layer_types_pattern_in_keys(self) -> None:
        """All 6 layer norms appear in keys for every layer."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        keys = set(model.state_dict().keys())

        for i in range(config.num_layers):
            # Core 4 norms
            assert f"decoder.layers.{i}.input_layernorm.weight" in keys
            assert f"decoder.layers.{i}.post_attention_layernorm.weight" in keys
            assert f"decoder.layers.{i}.pre_feedforward_layernorm.weight" in keys
            assert f"decoder.layers.{i}.post_feedforward_layernorm.weight" in keys

    def test_moe_keys_present(self) -> None:
        """MoE model has router and expert parameters."""
        config = self._make_small_config(enable_moe=True)

        with torch.device("meta"):
            model = create_gemma4_model(config)

        keys = set(model.state_dict().keys())

        # Every layer should have MoE keys
        for i in range(config.num_layers):
            assert f"decoder.layers.{i}.router.scale" in keys
            assert f"decoder.layers.{i}.router.per_expert_scale" in keys
            assert f"decoder.layers.{i}.experts.gate_up_proj" in keys
            assert f"decoder.layers.{i}.experts.down_proj" in keys
            assert f"decoder.layers.{i}.post_feedforward_layernorm_1.weight" in keys
            assert f"decoder.layers.{i}.pre_feedforward_layernorm_2.weight" in keys
            assert f"decoder.layers.{i}.post_feedforward_layernorm_2.weight" in keys

    def test_ple_keys_present(self) -> None:
        """PLE model has per-layer embedding parameters."""
        config = self._make_small_config_with_ple()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        keys = set(model.state_dict().keys())

        # Frontend PLE keys
        assert "decoder_frontend.embed_tokens_per_layer.weight" in keys
        assert "decoder_frontend.per_layer_model_projection.weight" in keys
        assert "decoder_frontend.per_layer_projection_norm.weight" in keys

        # Per-layer PLE keys
        for i in range(config.num_layers):
            assert f"decoder.layers.{i}.per_layer_input_gate.weight" in keys
            assert f"decoder.layers.{i}.per_layer_projection.weight" in keys
            assert f"decoder.layers.{i}.post_per_layer_input_norm.weight" in keys

    def test_layer_scalar_is_buffer(self) -> None:
        """layer_scalar should be a buffer, not a parameter."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        keys = set(model.state_dict().keys())

        for i in range(config.num_layers):
            assert f"decoder.layers.{i}.layer_scalar" in keys

        # Verify it's a buffer (not in parameters, but in state_dict)
        param_keys = {name for name, _ in model.named_parameters()}
        for i in range(config.num_layers):
            assert f"decoder.layers.{i}.layer_scalar" not in param_keys


# ---- Tests: HuggingFace Converter ----


class TestGemma4HuggingFaceConverter:
    """Tests for the _Gemma4HuggingFaceConverter class (fs2 → HF export)."""

    def _make_small_config(self, *, enable_moe: bool = False) -> Gemma4Config:
        """Create a tiny config for fast testing."""
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
            enable_moe=enable_moe,
            num_experts=4 if enable_moe else None,
            top_k_experts=2 if enable_moe else None,
            moe_intermediate_size=16 if enable_moe else None,
        )

    def _make_small_config_with_ple(self) -> Gemma4Config:
        """Create a tiny config with PLE enabled."""
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=128,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
        )

    # ---- to_hg_config tests ----

    def test_to_hg_config_returns_huggingface_config(self) -> None:
        """to_hg_config returns a HuggingFaceConfig instance."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        assert isinstance(hg_config, HuggingFaceConfig)

    def test_to_hg_config_class_names(self) -> None:
        """Config uses Gemma4TextConfig / Gemma4ForCausalLM."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Gemma4TextConfig"
        assert hg_config.arch == "Gemma4ForCausalLM"

    def test_to_hg_config_core_fields(self) -> None:
        """Core architecture fields are mapped correctly."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["hidden_size"] == 64
        assert data["vocab_size"] == 128
        assert data["num_hidden_layers"] == 6
        assert data["num_attention_heads"] == 4
        assert data["num_key_value_heads"] == 2
        assert data["head_dim"] == 16
        assert data["intermediate_size"] == 128
        assert data["tie_word_embeddings"] is True

    def test_to_hg_config_gemma4_specific_fields(self) -> None:
        """Gemma4-specific fields are present."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["global_head_dim"] == 32
        assert data["rope_theta_global"] == 1_000_000.0
        assert data["attention_k_eq_v"] is False
        assert data["num_kv_shared_layers"] == 0
        assert data["partial_rotary_factor"] == 0.25
        assert data["sliding_window"] == 32
        assert data["final_logit_softcapping"] == 30.0

    def test_to_hg_config_ple_fields(self) -> None:
        """PLE fields are present in HF config."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config_with_ple()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["hidden_size_per_layer_input"] == 16
        assert data["vocab_size_per_layer_input"] == 128

    def test_to_hg_config_moe_fields(self) -> None:
        """MoE fields are present when enable_moe=True."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config(enable_moe=True)
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["num_local_experts"] == 4
        assert data["num_experts_per_tok"] == 2
        assert data["moe_intermediate_size"] == 16

    def test_to_hg_config_no_moe_fields_when_disabled(self) -> None:
        """MoE fields are absent when enable_moe=False."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config(enable_moe=False)
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert "num_local_experts" not in data
        assert "num_experts_per_tok" not in data

    def test_to_hg_config_layer_types(self) -> None:
        """Layer types list is included in HF config."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert "layer_types" in data
        assert len(data["layer_types"]) == 6

    def test_to_hg_config_wrong_type_raises(self) -> None:
        """to_hg_config raises TypeError for wrong config type."""
        converter = _Gemma4HuggingFaceConverter()

        with pytest.raises(TypeError):
            converter.to_hg_config("not a config")

    @pytest.mark.parametrize(
        "config_fn,variant",
        [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ],
    )
    def test_to_hg_config_production_variants(self, config_fn, variant) -> None:
        """Production configs produce valid HF configs."""
        converter = _Gemma4HuggingFaceConverter()
        config = config_fn()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["hidden_size"] == config.model_dim
        assert data["num_hidden_layers"] == config.num_layers
        assert data["vocab_size"] == 262_144
        assert hg_config.kls_name == "Gemma4TextConfig"

    # ---- to_hg_state_dict tests ----

    def test_to_hg_state_dict_key_format(self) -> None:
        """Exported keys use model.* prefix (text-only format)."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        for key in hg_state_dict:
            assert key.startswith("model."), f"Expected model.* prefix: {key}"

    def test_to_hg_state_dict_tied_embeddings_no_lm_head(self) -> None:
        """With tied_embeddings, lm_head.weight is removed from export."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()  # tied_embeddings=True

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        assert "lm_head.weight" not in hg_state_dict

    def test_to_hg_state_dict_untied_has_lm_head(self) -> None:
        """Without tied_embeddings, lm_head.weight is preserved."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        config.tied_embeddings = False

        # Manually create a state dict with final_proj.proj.weight
        fs2_state_dict: dict[str, object] = {
            "final_proj.proj.weight": torch.empty(0),
            "decoder_frontend.embed.weight": torch.empty(0),
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        assert "lm_head.weight" in hg_state_dict

    def test_to_hg_state_dict_all_keys_converted(self) -> None:
        """All fs2 keys are converted to HF format (no unconverted keys)."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        # No key should retain fs2-style prefixes
        for key in hg_state_dict:
            assert not key.startswith("decoder."), f"Unconverted fs2 key: {key}"
            assert not key.startswith(
                "decoder_frontend."
            ), f"Unconverted fs2 key: {key}"
            assert not key.startswith("final_proj."), f"Unconverted fs2 key: {key}"

    def test_to_hg_state_dict_moe_keys(self) -> None:
        """MoE model exports router and expert keys correctly."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config(enable_moe=True)

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        # Check MoE keys exist with correct HF prefixes
        moe_keys = [k for k in hg_state_dict if "router" in k or "experts" in k]
        assert len(moe_keys) > 0, "No MoE keys in exported state dict"

        for key in moe_keys:
            assert key.startswith("model.layers."), f"MoE key wrong prefix: {key}"

    def test_to_hg_state_dict_ple_keys(self) -> None:
        """PLE model exports per-layer embedding keys correctly."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config_with_ple()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        # Frontend PLE keys
        ple_frontend_keys = [
            k
            for k in hg_state_dict
            if "embed_tokens_per_layer" in k
            or "per_layer_model_projection" in k
            or "per_layer_projection_norm" in k
        ]
        assert (
            len(ple_frontend_keys) >= 3
        ), f"Missing PLE frontend keys: {ple_frontend_keys}"

        # Per-layer PLE keys
        ple_layer_keys = [
            k
            for k in hg_state_dict
            if "per_layer_input_gate" in k
            or "per_layer_projection" in k
            and "model_projection" not in k
            or "post_per_layer_input_norm" in k
        ]
        assert len(ple_layer_keys) > 0, "No per-layer PLE keys in export"

    def test_to_hg_state_dict_wrong_type_raises(self) -> None:
        """to_hg_state_dict raises TypeError for wrong config type."""
        converter = _Gemma4HuggingFaceConverter()

        with pytest.raises(TypeError):
            converter.to_hg_state_dict({}, "not a config")

    # ---- Round-trip tests (fs2 → HF → fs2) via converter ----

    def test_text_key_map_round_trip(self) -> None:
        """fs2 → HF (text map) → fs2 is identity for basic model."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        # Forward: fs2 → HF via text key map
        reverse_map = create_reverse_key_map(_GEMMA4_TEXT_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)

        # Backward: HF → fs2 via text key map
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _GEMMA4_TEXT_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Text key map round-trip mismatch.\n"
            f"  Missing: {fs2_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - fs2_keys}"
        )

    def test_text_key_map_round_trip_moe(self) -> None:
        """fs2 → HF → fs2 round-trip for MoE model via text key map."""
        config = self._make_small_config(enable_moe=True)

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        reverse_map = create_reverse_key_map(_GEMMA4_TEXT_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _GEMMA4_TEXT_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"MoE text key map round-trip mismatch.\n"
            f"  Missing: {fs2_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - fs2_keys}"
        )

    def test_text_key_map_round_trip_ple(self) -> None:
        """fs2 → HF → fs2 round-trip for PLE model via text key map."""
        config = self._make_small_config_with_ple()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_keys = set(model.state_dict().keys())
        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        reverse_map = create_reverse_key_map(_GEMMA4_TEXT_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _GEMMA4_TEXT_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"PLE text key map round-trip mismatch.\n"
            f"  Missing: {fs2_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - fs2_keys}"
        )

    def test_converter_preserves_tensor_values(self) -> None:
        """Tensor values are preserved through to_hg_state_dict conversion."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        config.tied_embeddings = False

        weight = torch.randn(128, 64)
        fs2_state_dict: dict[str, object] = {
            "decoder_frontend.embed.weight": weight,
            "final_proj.proj.weight": weight.clone(),
        }

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        # Check the embedding weight is preserved
        assert "model.embed_tokens.weight" in hg_state_dict
        assert torch.equal(hg_state_dict["model.embed_tokens.weight"], weight)

    def test_converter_key_count_matches(self) -> None:
        """Number of keys in export matches fs2 state dict (minus tied weight)."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()  # tied_embeddings=True

        with torch.device("meta"):
            model = create_gemma4_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }
        fs2_count = len(fs2_state_dict)

        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)
        hg_count = len(hg_state_dict)

        # With tied embeddings, lm_head.weight is dropped from export,
        # so HF should have one fewer key than fs2 (the final_proj.proj.weight
        # is present in fs2 but maps to lm_head.weight which gets dropped).
        assert hg_count == fs2_count - 1, (
            f"Key count mismatch: fs2={fs2_count}, hg={hg_count} "
            f"(expected hg = fs2 - 1 for tied embeddings)"
        )

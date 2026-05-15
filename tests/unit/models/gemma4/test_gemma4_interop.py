# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Gemma 4 HuggingFace state-dict interop."""

from __future__ import annotations

import torch
from torch.testing import assert_close

from fairseq2.models.gemma4.config import Gemma4Config
from fairseq2.models.gemma4.factory import create_gemma4_model
from fairseq2.models.gemma4.interop import _HG_KEY_MAP, convert_gemma4_state_dict
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
        assert result["final_proj.proj.weight"] is result["decoder_frontend.embed.weight"]

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
                ("model.vision_tower.", "model.audio_tower.",
                 "model.embed_vision.", "model.embed_audio.",
                 "model.multi_modal_projector.")
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

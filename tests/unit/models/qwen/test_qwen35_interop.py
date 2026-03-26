# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Qwen 3.5 HuggingFace state-dict interop."""

from __future__ import annotations

import torch
from torch.testing import assert_close

from fairseq2.models.qwen.config import Qwen35Config, Qwen35MoeConfig
from fairseq2.models.qwen.factory import create_qwen35_model, create_qwen35_moe_model
from fairseq2.models.qwen.interop import (
    _QWEN35_HG_KEY_MAP,
    _QWEN35_RMSNORM_KEYS,
    _QWEN35_TEXT_KEY_MAP,
    _Qwen35HuggingFaceConverter,
    _Qwen35MoeHuggingFaceConverter,
    convert_qwen35_moe_state_dict,
    convert_qwen35_state_dict,
)
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map


class TestQwen35Interop:
    def _make_small_config(self) -> Qwen35Config:
        """Create a tiny config for fast testing."""
        config = Qwen35Config()
        config.model_dim = 64
        config.vocab_size = 128
        config.num_layers = 4  # 3 linear + 1 full attention
        config.num_attn_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.ffn_inner_dim = 128
        config.partial_rotary_factor = 0.25
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
        config.linear_key_head_dim = 8
        config.linear_value_head_dim = 8
        config.layer_types = None  # Reset so __post_init__ regenerates for num_layers=4
        config.__post_init__()
        return config

    def test_state_dict_key_round_trip(self) -> None:
        """fs2 keys -> HF keys -> fs2 keys should be identity."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_qwen35_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        # Convert to HF format using reverse key map
        reverse_map = create_reverse_key_map(_QWEN35_HG_KEY_MAP)
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)

        # Verify HF keys have expected prefixes
        for key in hg_state_dict:
            assert key.startswith(
                ("model.", "lm_head.")
            ), f"Unexpected HF key prefix: {key}"

        # Convert back to fs2 format
        rt_state_dict = convert_state_dict(dict(hg_state_dict), _QWEN35_HG_KEY_MAP)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Round-trip key mismatch.\n"
            f"  Missing in round-trip: {fs2_keys - rt_keys}\n"
            f"  Extra in round-trip:   {rt_keys - fs2_keys}"
        )

    def test_rmsnorm_weight_conversion(self) -> None:
        """RMSNorm weights get +1.0 added during conversion."""
        config = self._make_small_config()

        # Simulate HF state dict with zero-init RMSNorm weights
        hf_state_dict: dict[str, object] = {}
        for i in range(config.num_layers):
            hf_state_dict[f"model.layers.{i}.input_layernorm.weight"] = torch.zeros(
                config.model_dim
            )
            hf_state_dict[
                f"model.layers.{i}.post_attention_layernorm.weight"
            ] = torch.zeros(config.model_dim)
        hf_state_dict["model.norm.weight"] = torch.zeros(config.model_dim)
        hf_state_dict["model.embed_tokens.weight"] = torch.zeros(
            config.vocab_size, config.model_dim
        )
        hf_state_dict["lm_head.weight"] = torch.zeros(
            config.vocab_size, config.model_dim
        )

        converted = convert_qwen35_state_dict(dict(hf_state_dict), config)

        # All layer norm weights should now be 1.0 (0.0 + 1.0)
        for key in converted:
            if any(key.endswith(s) for s in _QWEN35_RMSNORM_KEYS):
                weight = converted[key]
                assert isinstance(weight, torch.Tensor)
                assert_close(weight, torch.ones_like(weight))

    def test_gdn_norm_weight_not_converted(self) -> None:
        """GatedDeltaNet internal norm weights should NOT get +1.0."""
        config = self._make_small_config()

        # Simulate HF state dict with GDN norm weight
        hf_state_dict: dict[str, object] = {
            "model.embed_tokens.weight": torch.zeros(1)
        }
        hf_state_dict["model.layers.0.linear_attn.norm.weight"] = (
            torch.ones(config.linear_value_head_dim) * 0.5
        )

        converted = convert_qwen35_state_dict(dict(hf_state_dict), config)

        # The GDN norm maps to linear_attn.norm.inner_norm.weight
        gdn_key = "decoder.layers.0.linear_attn.norm.inner_norm.weight"
        if gdn_key in converted:
            # Should still be 0.5, NOT 1.5
            assert_close(
                converted[gdn_key],
                torch.ones(config.linear_value_head_dim) * 0.5,
            )

    def test_tied_embeddings_hf_no_lm_head(self) -> None:
        """HF checkpoint with tied_embeddings has no lm_head.weight.

        Safetensors deduplicates shared tensors, so for models with
        tie_word_embeddings=True the checkpoint only contains
        model.embed_tokens.weight. The converter must create
        final_proj.weight from it.
        """
        config = self._make_small_config()
        config.tied_embeddings = True

        weight = torch.randn(config.vocab_size, config.model_dim)
        hf_state_dict: dict[str, object] = {
            "model.embed_tokens.weight": weight,
            "model.norm.weight": torch.zeros(config.model_dim),
        }

        result = convert_qwen35_state_dict(dict(hf_state_dict), config)

        assert "decoder_frontend.embed.weight" in result
        assert "final_proj.weight" in result
        assert result["final_proj.weight"] is result["decoder_frontend.embed.weight"]

    def test_layer_types_are_correct(self) -> None:
        """Verify layer_types pattern: 3 linear, 1 full, repeating."""
        config = self._make_small_config()
        assert config.layer_types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]


class TestQwen35HuggingFaceConverter:
    """Tests for _Qwen35HuggingFaceConverter."""

    def _make_small_config(self) -> Qwen35Config:
        config = Qwen35Config()
        config.model_dim = 64
        config.vocab_size = 128
        config.num_layers = 4
        config.num_attn_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.ffn_inner_dim = 128
        config.partial_rotary_factor = 0.25
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
        config.linear_key_head_dim = 8
        config.linear_value_head_dim = 8
        config.layer_types = None
        config.__post_init__()
        return config

    def test_to_hg_config(self) -> None:
        """to_hg_config maps Qwen35Config fields to HF config dict."""
        config = self._make_small_config()
        converter = _Qwen35HuggingFaceConverter()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Qwen3_5TextConfig"
        assert hg_config.arch == "Qwen3_5ForCausalLM"

        data = hg_config.data
        assert data["hidden_size"] == config.model_dim
        assert data["max_position_embeddings"] == config.max_seq_len
        assert data["vocab_size"] == config.vocab_size
        assert data["tie_word_embeddings"] == config.tied_embeddings
        assert data["num_hidden_layers"] == config.num_layers
        assert data["num_attention_heads"] == config.num_attn_heads
        assert data["num_key_value_heads"] == config.num_key_value_heads
        assert data["head_dim"] == config.head_dim
        assert data["intermediate_size"] == config.ffn_inner_dim
        assert data["partial_rotary_factor"] == config.partial_rotary_factor
        assert data["rope_theta"] == config.rope_theta
        assert data["full_attention_interval"] == config.full_attention_interval
        assert data["linear_conv_kernel_dim"] == config.linear_conv_kernel_dim
        assert data["linear_key_head_dim"] == config.linear_key_head_dim
        assert data["linear_value_head_dim"] == config.linear_value_head_dim
        assert data["linear_num_key_heads"] == config.linear_num_key_heads
        assert data["linear_num_value_heads"] == config.linear_num_value_heads

    def test_state_dict_round_trip(self) -> None:
        """State dict keys survive a fs2 -> HF -> fs2 round trip."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_qwen35_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        converter = _Qwen35HuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)
        hg_keys = set(hg_state_dict.keys())

        for key in hg_keys:
            assert key.startswith(
                ("model.", "lm_head.")
            ), f"Unexpected HF key prefix: {key}"

        # Round-trip: HF -> fs2
        rt_state_dict = convert_qwen35_state_dict(dict(hg_state_dict), config)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Round-trip key mismatch.\n"
            f"  Missing in round-trip: {fs2_keys - rt_keys}\n"
            f"  Extra in round-trip:   {rt_keys - fs2_keys}"
        )

    def test_rmsnorm_weight_reversed(self) -> None:
        """to_hg_state_dict subtracts 1.0 from RMSNorm weights."""
        config = self._make_small_config()

        # Build a fs2 state dict with RMSNorm weights = 1.0 (standard init)
        fs2_state_dict: dict[str, object] = {}
        for i in range(config.num_layers):
            fs2_state_dict[f"decoder.layers.{i}.self_attn_layer_norm.weight"] = (
                torch.ones(config.model_dim)
            )
            fs2_state_dict[f"decoder.layers.{i}.ffn_layer_norm.weight"] = torch.ones(
                config.model_dim
            )
        fs2_state_dict["decoder.layer_norm.weight"] = torch.ones(config.model_dim)

        converter = _Qwen35HuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        # HF weights should be 0.0 (1.0 - 1.0)
        for key in hg_state_dict:
            if key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight")):
                weight = hg_state_dict[key]
                assert isinstance(weight, torch.Tensor)
                assert_close(weight, torch.zeros_like(weight))

    def test_tied_embeddings_removes_lm_head(self) -> None:
        """to_hg_state_dict removes lm_head.weight when tied_embeddings=True."""
        config = self._make_small_config()
        config.tied_embeddings = True

        with torch.device("meta"):
            model = create_qwen35_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        converter = _Qwen35HuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        assert "lm_head.weight" not in hg_state_dict
        assert "model.embed_tokens.weight" in hg_state_dict

    def test_tied_embeddings_deduped_final_proj_only(self) -> None:
        """When safetensors deduplicates tied weights and only final_proj.weight
        survives, convert_qwen35_state_dict should still reconstruct both keys."""
        config = self._make_small_config()
        config.tied_embeddings = True

        weight = torch.randn(config.vocab_size, config.model_dim)
        state_dict: dict[str, object] = {"final_proj.weight": weight}

        result = convert_qwen35_state_dict(dict(state_dict), config)

        assert "decoder_frontend.embed.weight" in result
        assert result["decoder_frontend.embed.weight"] is weight


class TestQwen35MoeHuggingFaceConverter:
    """Tests for _Qwen35MoeHuggingFaceConverter."""

    def _make_small_moe_config(self) -> Qwen35MoeConfig:
        config = Qwen35MoeConfig()
        config.model_dim = 64
        config.vocab_size = 128
        config.num_layers = 4
        config.num_attn_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.ffn_inner_dim = 128
        config.partial_rotary_factor = 0.25
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
        config.linear_key_head_dim = 8
        config.linear_value_head_dim = 8
        config.num_experts = 4
        config.num_experts_per_tok = 2
        config.moe_intermediate_size = 32
        config.shared_expert_intermediate_size = 32
        config.layer_types = None
        config.__post_init__()
        return config

    def test_to_hg_config(self) -> None:
        """to_hg_config maps Qwen35MoeConfig fields including MoE-specific ones."""
        config = self._make_small_moe_config()
        converter = _Qwen35MoeHuggingFaceConverter()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Qwen3_5TextConfig"
        assert hg_config.arch == "Qwen3_5MoeForCausalLM"

        data = hg_config.data
        assert data["hidden_size"] == config.model_dim
        assert data["num_experts"] == config.num_experts
        assert data["num_experts_per_tok"] == config.num_experts_per_tok
        assert data["moe_intermediate_size"] == config.moe_intermediate_size
        assert data["shared_expert_intermediate_size"] == config.shared_expert_intermediate_size
        assert data["router_aux_loss_coef"] == config.router_aux_loss_coef

    def test_state_dict_round_trip(self) -> None:
        """MoE state dict keys survive a fs2 -> HF -> fs2 round trip."""
        config = self._make_small_moe_config()

        with torch.device("meta"):
            model = create_qwen35_moe_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        converter = _Qwen35MoeHuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)
        hg_keys = set(hg_state_dict.keys())

        for key in hg_keys:
            assert key.startswith(
                ("model.", "lm_head.")
            ), f"Unexpected HF key prefix: {key}"

        # Round-trip: HF -> fs2
        rt_state_dict = convert_qwen35_moe_state_dict(dict(hg_state_dict), config)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Round-trip key mismatch.\n"
            f"  Missing in round-trip: {fs2_keys - rt_keys}\n"
            f"  Extra in round-trip:   {rt_keys - fs2_keys}"
        )


class TestVlCheckpointHandling:
    """Tests for multimodal (VL) checkpoint handling.

    Qwen 3.5 checkpoints on HuggingFace Hub are multimodal models where the
    text decoder lives under ``model.language_model.*`` with additional
    ``model.visual.*`` and ``mtp.*`` keys.  The converter handles both formats
    via ``_expand_with_language_model_prefix`` and explicit filtering.
    """

    def _make_small_config(self) -> Qwen35Config:
        config = Qwen35Config()
        config.model_dim = 64
        config.vocab_size = 128
        config.num_layers = 4
        config.num_attn_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.ffn_inner_dim = 128
        config.partial_rotary_factor = 0.25
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
        config.linear_key_head_dim = 8
        config.linear_value_head_dim = 8
        config.tied_embeddings = True
        config.layer_types = None
        config.__post_init__()
        return config

    def test_key_map_has_language_model_variants(self) -> None:
        """_QWEN35_HG_KEY_MAP includes both model.* and model.language_model.* patterns."""
        text_only_count = len(_QWEN35_TEXT_KEY_MAP)
        full_count = len(_QWEN35_HG_KEY_MAP)
        # model.* patterns get duplicated; lm_head.* does not
        model_prefix_count = sum(
            1 for k in _QWEN35_TEXT_KEY_MAP if k.startswith(r"^model\.")
        )
        assert full_count == text_only_count + model_prefix_count

    def test_language_model_prefix_keys_convert(self) -> None:
        """model.language_model.X keys are correctly converted to fs2 keys."""
        state_dict: dict[str, object] = {
            "model.language_model.embed_tokens.weight": torch.empty(0),
            "model.language_model.layers.0.input_layernorm.weight": torch.empty(0),
            "model.language_model.norm.weight": torch.empty(0),
        }
        result = convert_state_dict(state_dict, _QWEN35_HG_KEY_MAP)
        assert "decoder_frontend.embed.weight" in result
        assert "decoder.layers.0.self_attn_layer_norm.weight" in result
        assert "decoder.layer_norm.weight" in result

    def test_visual_and_mtp_keys_filtered(self) -> None:
        """model.visual.* and mtp.* keys are filtered by convert_qwen35_state_dict."""
        config = self._make_small_config()
        state_dict: dict[str, object] = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.model_dim
            ),
            "model.language_model.norm.weight": torch.zeros(config.model_dim),
            "model.visual.blocks.0.attn.proj.weight": torch.empty(0),
            "model.visual.patch_embed.proj.weight": torch.empty(0),
            "mtp.fc.weight": torch.empty(0),
            "mtp.layers.0.mlp.gate_proj.weight": torch.empty(0),
        }
        result = convert_qwen35_state_dict(dict(state_dict), config)
        for key in result:
            assert not key.startswith(("model.visual.", "mtp.")), (
                f"Unexpected key not filtered: {key}"
            )

    def test_text_only_format_still_works(self) -> None:
        """model.layers.* (text-only format) is still handled correctly."""
        config = self._make_small_config()
        state_dict: dict[str, object] = {
            "model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.model_dim
            ),
            "model.norm.weight": torch.zeros(config.model_dim),
        }
        result = convert_qwen35_state_dict(dict(state_dict), config)
        assert "decoder_frontend.embed.weight" in result
        assert "decoder.layer_norm.weight" in result

    def test_end_to_end_vl_checkpoint(self) -> None:
        """Full VL checkpoint → convert_qwen35_state_dict produces correct keys."""
        config = self._make_small_config()

        with torch.device("meta"):
            model = create_qwen35_model(config)
        model_keys = set(model.state_dict().keys())

        # Build a text-only HF state dict, then add VL prefix + extra modalities
        reverse_map = create_reverse_key_map(_QWEN35_TEXT_KEY_MAP)
        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in model_keys}
        hg_state_dict = convert_state_dict(fs2_state_dict, reverse_map)

        # Add model.language_model. prefix (simulating VL checkpoint)
        vl_state_dict: dict[str, object] = {}
        for k, v in hg_state_dict.items():
            if k.startswith("model."):
                vl_state_dict["model.language_model." + k[len("model."):]] = v
            else:
                vl_state_dict[k] = v
        # Add visual/mtp keys
        vl_state_dict["model.visual.blocks.0.attn.proj.weight"] = torch.empty(0)
        vl_state_dict["mtp.fc.weight"] = torch.empty(0)

        # Convert back — should match model keys
        result = convert_qwen35_state_dict(dict(vl_state_dict), config)
        result_keys = set(result.keys())

        assert model_keys == result_keys, (
            f"VL round-trip key mismatch.\n"
            f"  Missing: {model_keys - result_keys}\n"
            f"  Extra:   {result_keys - model_keys}"
        )

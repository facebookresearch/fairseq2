# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Qwen 3.5 HuggingFace state-dict interop."""

from __future__ import annotations

import torch
from torch.testing import assert_close

from fairseq2.models.qwen.config import Qwen35Config
from fairseq2.models.qwen.factory import create_qwen35_model
from fairseq2.models.qwen.interop import (
    _QWEN35_HG_KEY_MAP,
    _QWEN35_RMSNORM_KEYS,
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

    def test_layer_types_are_correct(self) -> None:
        """Verify layer_types pattern: 3 linear, 1 full, repeating."""
        config = self._make_small_config()
        assert config.layer_types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH config and hybrid pattern parsing."""

from __future__ import annotations

import pytest

from fairseq2.models.nemotron.config import (
    NemotronHConfig,
    parse_hybrid_pattern,
)


class TestParseHybridPattern:
    def test_basic_pattern(self) -> None:
        result = parse_hybrid_pattern("M E A")
        assert result == ["mamba", "moe", "attention"]

    def test_full_52_layer_pattern(self) -> None:
        config = NemotronHConfig()
        layer_types = config.layer_types
        assert len(layer_types) == 52

    def test_layer_counts(self) -> None:
        config = NemotronHConfig()
        layer_types = config.layer_types
        mamba_count = sum(1 for t in layer_types if t == "mamba")
        moe_count = sum(1 for t in layer_types if t == "moe")
        attn_count = sum(1 for t in layer_types if t == "attention")
        assert mamba_count == 23
        assert moe_count == 23
        assert attn_count == 6

    def test_attention_positions(self) -> None:
        """Attention layers should be at positions 5,12,19,26,33,42."""
        config = NemotronHConfig()
        layer_types = config.layer_types
        attn_positions = [i for i, t in enumerate(layer_types) if t == "attention"]
        assert attn_positions == [5, 12, 19, 26, 33, 42]

    def test_invalid_character(self) -> None:
        with pytest.raises(ValueError, match="Invalid block type"):
            parse_hybrid_pattern("M E X")

    def test_empty_pattern(self) -> None:
        result = parse_hybrid_pattern("")
        assert result == []


class TestNemotronHConfig:
    def test_default_config(self) -> None:
        config = NemotronHConfig()
        assert config.model_dim == 2688
        assert config.vocab_size == 131_072
        assert config.num_layers == 52
        assert config.num_experts == 128
        assert config.num_experts_per_tok == 6
        assert config.mamba_num_heads == 64
        assert config.mamba_head_dim == 64
        assert config.ssm_state_size == 128
        assert config.tied_embeddings is False
        assert config.rms_norm_eps == 1e-5

    def test_derived_dims(self) -> None:
        config = NemotronHConfig()
        assert config.mamba_intermediate_size == 4096  # 64*64
        assert config.mamba_conv_dim == 6144  # 4096 + 2*8*128
        assert config.mamba_projection_size == 10304  # 4096 + 6144 + 64

    def test_validation_passes_default(self) -> None:
        config = NemotronHConfig()
        config.validate()

    def test_validation_pattern_length_mismatch(self) -> None:
        config = NemotronHConfig()
        config.hybrid_override_pattern = "M E A"  # 3, not 52
        with pytest.raises(ValueError, match="does not match num_layers"):
            config.validate()

    def test_validation_missing_mamba(self) -> None:
        config = NemotronHConfig()
        config.num_layers = 2
        config.hybrid_override_pattern = "E A"
        with pytest.raises(ValueError, match="at least one Mamba2"):
            config.validate()

    def test_validation_missing_moe(self) -> None:
        config = NemotronHConfig()
        config.num_layers = 2
        config.hybrid_override_pattern = "M A"
        with pytest.raises(ValueError, match="at least one MoE"):
            config.validate()

    def test_validation_missing_attention(self) -> None:
        config = NemotronHConfig()
        config.num_layers = 2
        config.hybrid_override_pattern = "M E"
        with pytest.raises(ValueError, match="at least one Attention"):
            config.validate()

    def test_custom_config(self) -> None:
        config = NemotronHConfig()
        config.model_dim = 512
        config.num_layers = 6
        config.hybrid_override_pattern = "M E M A E M"
        config.validate()
        assert len(config.layer_types) == 6

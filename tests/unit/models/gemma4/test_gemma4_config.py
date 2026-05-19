# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Gemma 4 configuration and model factory."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.gemma4.config import (
    Gemma4Config,
    _compute_layer_types,
    get_gemma4_26b_a4b_config,
    get_gemma4_31b_config,
    get_gemma4_e4b_config,
    get_kv_projection_role,
)
from fairseq2.models.gemma4.factory import create_gemma4_model
from fairseq2.models.gemma4.model import Gemma4Model
from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.nn import BatchLayout
from tests.common import device


class TestGemma4Config:
    """Test Gemma4Config creation and helper functions."""

    def test_e4b_config_defaults(self) -> None:
        """E4B config has expected dimensions."""
        config = get_gemma4_e4b_config()
        assert config.model_dim == 2560
        assert config.num_layers == 42
        assert config.num_attn_heads == 8
        assert config.num_key_value_heads == 2
        assert config.head_dim == 256
        assert config.global_head_dim == 512
        assert config.ffn_inner_dim == 10_240
        assert config.sliding_window == 512
        assert config.attention_k_eq_v is False
        assert config.num_kv_shared_layers == 18
        assert config.hidden_size_per_layer_input == 256
        assert config.has_ple is True
        assert config.enable_moe is False

    def test_31b_config_defaults(self) -> None:
        """31B config has expected dimensions."""
        config = get_gemma4_31b_config()
        assert config.model_dim == 5376
        assert config.num_layers == 60
        assert config.attention_k_eq_v is True
        assert config.num_kv_shared_layers == 0
        assert config.hidden_size_per_layer_input == 0
        assert config.has_ple is False
        assert config.enable_moe is False

    def test_26b_a4b_config_defaults(self) -> None:
        """26B-A4B config has MoE enabled."""
        config = get_gemma4_26b_a4b_config()
        assert config.model_dim == 2816
        assert config.num_layers == 30
        assert config.attention_k_eq_v is True
        assert config.enable_moe is True
        assert config.num_experts == 128
        assert config.top_k_experts == 8
        assert config.moe_intermediate_size == 704
        assert config.has_ple is False

    def test_layer_types_auto_computed(self) -> None:
        """layer_types is auto-computed from 5:1 pattern when empty."""
        config = Gemma4Config(num_layers=12)
        assert len(config.layer_types) == 12
        # Every 6th layer (1-indexed) should be full_attention
        for i in range(12):
            if (i + 1) % 6 == 0 or i == 11:  # Last layer always full
                assert config.layer_types[i] == "full_attention", f"Layer {i}"
            else:
                assert config.layer_types[i] == "sliding_attention", f"Layer {i}"


class TestComputeLayerTypes:
    """Test the 5:1 sliding:full attention pattern via _compute_layer_types."""

    def test_basic_pattern(self) -> None:
        """Layers 5, 11, ... are full attention (0-indexed)."""
        num_layers = 42
        layer_types = _compute_layer_types(num_layers)
        full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        # Every 6th layer (1-indexed: 6, 12, 18, ...) maps to 0-indexed: 5, 11, 17, ...
        expected = [i for i in range(num_layers) if (i + 1) % 6 == 0]
        # Last layer is always full
        if (num_layers - 1) not in expected:
            expected.append(num_layers - 1)
        assert full_layers == sorted(expected)

    def test_last_layer_always_full(self) -> None:
        """The last layer is always full attention, regardless of pattern."""
        for num_layers in [6, 12, 30, 42, 60]:
            assert _compute_layer_types(num_layers)[-1] == "full_attention"

    def test_first_layer_sliding(self) -> None:
        """Layer 0 is always sliding attention."""
        for num_layers in [6, 12, 42]:
            assert _compute_layer_types(num_layers)[0] == "sliding_attention"

    @pytest.mark.parametrize("num_layers", [6, 12, 30, 42, 60])
    def test_full_attention_count(self, num_layers: int) -> None:
        """Count of full attention layers matches expectation."""
        layer_types = _compute_layer_types(num_layers)
        full_count = sum(1 for t in layer_types if t == "full_attention")
        # Approximately 1/6 of layers + possibly last layer adjustment
        expected_from_pattern = num_layers // 6
        # Last layer may add one more if not already in pattern
        if num_layers % 6 != 0:
            expected_from_pattern += 1  # Last layer forced full
        assert full_count == expected_from_pattern


class TestGetKvProjectionRole:
    """Test KV projection role assignment."""

    def test_no_sharing(self) -> None:
        """All layers are NONE when num_kv_shared_layers=0."""
        num_layers = 12
        layer_types = _compute_layer_types(num_layers)
        for i in range(num_layers):
            role = get_kv_projection_role(
                i, layer_types[i], num_layers, 0, layer_types
            )
            assert role == KVProjectionRole.NONE

    def test_e4b_sharing_pattern(self) -> None:
        """E4B has 18 shared layers (last 18 are CONSUMER)."""
        num_layers = 42
        num_shared = 18
        layer_types = _compute_layer_types(num_layers)

        roles = [
            get_kv_projection_role(i, layer_types[i], num_layers, num_shared, layer_types)
            for i in range(num_layers)
        ]

        # Last 18 layers (24-41) should be CONSUMER
        first_shared = num_layers - num_shared  # 24
        for i in range(first_shared, num_layers):
            assert roles[i] == KVProjectionRole.CONSUMER, f"Layer {i} should be CONSUMER"

        # There should be at least one SOURCE layer for each attention type
        source_layers = [i for i, r in enumerate(roles) if r == KVProjectionRole.SOURCE]
        assert len(source_layers) >= 1, "Must have at least one SOURCE layer"

        # SOURCE layers must be before the sharing boundary
        for i in source_layers:
            assert i < first_shared, f"SOURCE layer {i} must be before boundary {first_shared}"


class TestGemma4ModelFactory:
    """Test model creation with a small config."""

    def _make_small_config(self) -> Gemma4Config:
        """Create a tiny Gemma 4 config for fast testing."""
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,  # 5 sliding + 1 full (last)
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,  # PLE disabled for simplicity
            final_logit_soft_cap=None,
            tied_embeddings=True,
        )

    def test_create_model_on_meta_device(self) -> None:
        """Model creation on meta device succeeds."""
        config = self._make_small_config()
        with torch.device("meta"):
            model = create_gemma4_model(config)
        assert isinstance(model, Gemma4Model)

    def test_model_forward_shape(self) -> None:
        """Model forward produces (B, S, V) logits."""
        config = self._make_small_config()
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 8, 128)

    def test_model_with_ple(self) -> None:
        """Model with PLE enabled produces correct shape."""
        config = self._make_small_config()
        config.hidden_size_per_layer_input = 16
        config.vocab_size_per_layer_input = 128
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 8, 128)

    def test_model_with_moe(self) -> None:
        """Model with MoE enabled produces correct shape."""
        config = self._make_small_config()
        config.enable_moe = True
        config.num_experts = 4
        config.top_k_experts = 2
        config.moe_intermediate_size = 32
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 8, 128)

    def test_model_has_hybrid_layers(self) -> None:
        """Model should have both sliding and full attention layers."""
        config = self._make_small_config()
        with torch.device("meta"):
            model = create_gemma4_model(config)

        layer_types = config.layer_types
        assert "sliding_attention" in layer_types
        assert "full_attention" in layer_types
        # Last layer should be full_attention
        assert layer_types[-1] == "full_attention"

    def test_model_with_kv_sharing(self) -> None:
        """Model with KV sharing produces correct shape."""
        config = self._make_small_config()
        # Need enough layers so that both a full_attention SOURCE and a
        # full_attention CONSUMER exist.  With num_layers=6 there is only
        # one full_attention layer (the last), which becomes a CONSUMER —
        # no global SOURCE.  With 12 layers, layer 5 is full_attention and
        # sits before the sharing boundary, acting as the global SOURCE.
        config.num_layers = 12
        config.layer_types = []  # force recomputation
        config.__post_init__()
        config.num_kv_shared_layers = 2  # Last 2 layers share KV
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 8, 128)

    def test_model_with_softcapped_projection(self) -> None:
        """Model with softcapping on final logits produces bounded outputs."""
        config = self._make_small_config()
        config.final_logit_soft_cap = 30.0
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 8, 128)
        # Softcapped logits should be bounded by [-cap, cap]
        assert logits.abs().max() <= 30.0 + 1e-3

    def test_model_loss_computation(self) -> None:
        """Model forward with targets returns a scalar loss."""
        config = self._make_small_config()
        model = create_gemma4_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8), device=device)
        targets = torch.randint(0, 128, (1, 8), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            loss = model(input_ids, layout, targets)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Cross-entropy loss should be positive

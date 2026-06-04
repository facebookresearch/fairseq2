# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH decoder layer (3-way hybrid) and full model."""

from __future__ import annotations

import torch

from fairseq2.models.nemotron.config import NemotronHConfig
from fairseq2.models.nemotron.decoder_layer import NemotronHBlock
from fairseq2.models.nemotron.factory import NemotronHFactory
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout


def _make_tiny_config() -> NemotronHConfig:
    """Create a minimal config for fast testing."""
    config = NemotronHConfig()
    config.model_dim = 128
    config.vocab_size = 1000
    config.num_layers = 6
    config.hybrid_override_pattern = "M E M A E M"
    config.max_seq_len = 256
    config.num_attn_heads = 4
    config.num_key_value_heads = 2
    config.attn_head_dim = 32
    config.mamba_num_heads = 4
    config.mamba_head_dim = 32
    config.ssm_state_size = 16
    config.mamba_n_groups = 2
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 64
    config.shared_expert_intermediate_size = 128
    return config


class TestNemotronHBlock:
    def test_mamba_block(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        block = factory.create_decoder_layer(0, "mamba")
        assert isinstance(block, NemotronHBlock)
        assert block.block_type == "mamba"

        x = torch.randn(2, 8, 128)
        layout = BatchLayout.of(torch.zeros(2, 8, dtype=torch.long))
        cache = AttentionBiasCache()
        out = block(x, layout, cache)
        assert out.shape == (2, 8, 128)

    def test_moe_block(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        block = factory.create_decoder_layer(1, "moe")
        assert isinstance(block, NemotronHBlock)
        assert block.block_type == "moe"

        x = torch.randn(2, 8, 128)
        layout = BatchLayout.of(torch.zeros(2, 8, dtype=torch.long))
        cache = AttentionBiasCache()
        out = block(x, layout, cache)
        assert out.shape == (2, 8, 128)

    def test_attention_block(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        block = factory.create_decoder_layer(3, "attention")
        assert isinstance(block, NemotronHBlock)
        assert block.block_type == "attention"

        x = torch.randn(2, 8, 128)
        layout = BatchLayout.of(torch.zeros(2, 8, dtype=torch.long))
        cache = AttentionBiasCache()
        out = block(x, layout, cache)
        assert out.shape == (2, 8, 128)

    def test_residual_connection(self) -> None:
        """Output should not be identical to input (mixer changes it)."""
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        block = factory.create_decoder_layer(0, "mamba")

        x = torch.randn(2, 8, 128)
        layout = BatchLayout.of(torch.zeros(2, 8, dtype=torch.long))
        cache = AttentionBiasCache()

        with torch.no_grad():
            out = block(x, layout, cache)

        # Output should include residual (not zero, not identical to input)
        assert not torch.allclose(out, x, atol=1e-3)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-3)


class TestNemotronHFactory:
    def test_create_model(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()
        assert model is not None

    def test_model_forward(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            output = model(input_ids, layout)

        assert output.shape == (2, 16, config.vocab_size)

    def test_model_training_mode(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        layout = BatchLayout.of(input_ids)

        loss = model(input_ids, layout, targets=targets)
        assert loss.dim() == 0 or loss.numel() == 1  # scalar loss

    def test_layer_pattern_matches(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        # Check that the layer types match the pattern
        expected_types = ["mamba", "moe", "mamba", "attention", "moe", "mamba"]
        for idx, (layer, expected) in enumerate(
            zip(model.decoder.layers, expected_types)
        ):
            assert isinstance(layer, NemotronHBlock)
            assert layer.block_type == expected, (
                f"Layer {idx}: expected {expected}, got {layer.block_type}"
            )

    def test_parameter_count_reasonable(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        total_params = sum(p.numel() for p in model.parameters())
        # Tiny model should have reasonable params (not 0, not huge)
        assert 100_000 < total_params < 10_000_000

    def test_no_tied_embeddings(self) -> None:
        config = _make_tiny_config()
        config.tied_embeddings = False
        factory = NemotronHFactory(config)
        model = factory.create_model()

        # Check that embed and final_proj weights are different objects
        embed_weight = model.decoder_frontend.embed.weight
        final_weight = model.final_proj.weight
        assert embed_weight.data_ptr() != final_weight.data_ptr()

    def test_gradient_flow_through_model(self) -> None:
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        layout = BatchLayout.of(input_ids)

        loss = model(input_ids, layout, targets=targets)
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters have non-zero gradients!"


class TestNemotronHInterop:
    def test_key_mapping_basic(self) -> None:
        """Test that HF->FS2 key mapping works for a small model."""
        config = _make_tiny_config()
        factory = NemotronHFactory(config)
        model = factory.create_model()

        # Get FS2 state dict
        fs2_sd = model.state_dict()

        # Verify key structure
        assert "decoder_frontend.embed.weight" in fs2_sd
        assert "decoder.layer_norm.weight" in fs2_sd
        assert "final_proj.weight" in fs2_sd

        # Check layer keys exist
        assert "decoder.layers.0.norm.weight" in fs2_sd
        assert "decoder.layers.0.mixer.in_proj.weight" in fs2_sd  # Mamba layer
        assert "decoder.layers.1.mixer.gate.weight" in fs2_sd  # MoE layer

    def test_convert_hf_to_fs2(self) -> None:
        """Test converting a synthetic HF state dict to FS2 format."""
        from fairseq2.models.nemotron.interop import convert_nemotron_h_state_dict

        config = _make_tiny_config()

        # Create a fake HF state dict
        hf_sd: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.randn(1000, 128),
            "language_model.backbone.norm_f.weight": torch.randn(128),
            "language_model.lm_head.weight": torch.randn(1000, 128),
            "language_model.backbone.layers.0.norm.weight": torch.randn(128),
            "language_model.backbone.layers.0.mixer.in_proj.weight": torch.randn(
                config.mamba_projection_size, 128
            ),
        }

        fs2_sd = convert_nemotron_h_state_dict(hf_sd, config)

        assert "decoder_frontend.embed.weight" in fs2_sd
        assert "decoder.layer_norm.weight" in fs2_sd
        assert "final_proj.weight" in fs2_sd
        assert "decoder.layers.0.norm.weight" in fs2_sd
        assert "decoder.layers.0.mixer.in_proj.weight" in fs2_sd

    def test_skip_multimodal_keys(self) -> None:
        """Multimodal keys should be filtered out in Phase 1."""
        from fairseq2.models.nemotron.interop import convert_nemotron_h_state_dict

        config = _make_tiny_config()

        hf_sd: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.randn(1000, 128),
            "language_model.backbone.norm_f.weight": torch.randn(128),
            "language_model.lm_head.weight": torch.randn(1000, 128),
            "vision_model.radio_model.some_key": torch.randn(10),
            "mlp1.0.weight": torch.randn(10, 10),
            "sound_encoder.encoder.some_key": torch.randn(10),
            "sound_projection.norm.weight": torch.randn(10),
        }

        fs2_sd = convert_nemotron_h_state_dict(hf_sd, config)

        # Multimodal keys should not be present
        for key in fs2_sd:
            assert not key.startswith("vision_model.")
            assert not key.startswith("mlp1.")
            assert not key.startswith("sound_encoder.")
            assert not key.startswith("sound_projection.")

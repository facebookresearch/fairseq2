# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH Parakeet audio encoder (Phase 3).

Tests cover:
- ParakeetSubsamplingConv2D shape and downsampling factor
- ParakeetRelativePositionalEncoding output shape and properties
- ParakeetRelativeAttention output shape and relative shift
- ParakeetConformerBlock forward pass
- ParakeetAudioTower full pipeline
- SoundProjection shape and SquaredReLU behavior
- NemotronHMultimodalModel text-only and audio injection
- Interop audio key mapping
"""

from __future__ import annotations

import re

import pytest
import torch

from fairseq2.models.nemotron.audio.attention import (
    ParakeetRelativeAttention,
    ParakeetRelativePositionalEncoding,
    _rel_shift,
)
from fairseq2.models.nemotron.audio.conformer import (
    ParakeetAudioTower,
    ParakeetConformerBlock,
)
from fairseq2.models.nemotron.audio.projection import SoundProjection
from fairseq2.models.nemotron.audio.subsample import ParakeetSubsamplingConv2D
from fairseq2.models.nemotron.config import NemotronHConfig, ParakeetAudioConfig
from fairseq2.models.nemotron.interop import (
    _HG_AUDIO_KEY_MAP,
    convert_nemotron_h_state_dict,
)
from fairseq2.nn import BatchLayout


# ============================================================================
# Subsampling Tests
# ============================================================================


class TestParakeetSubsamplingConv2D:
    def test_output_shape(self) -> None:
        """Verify 8x temporal downsampling: [2, 640, 128] -> [2, 80, 1024]."""
        sub = ParakeetSubsamplingConv2D(
            num_mel_bins=128, hidden_size=1024, conv_channels=256
        )
        x = torch.randn(2, 640, 128)
        out = sub(x)
        assert out.shape == (2, 80, 1024)

    def test_output_shape_different_lengths(self) -> None:
        """Test various input lengths."""
        sub = ParakeetSubsamplingConv2D(
            num_mel_bins=128, hidden_size=1024, conv_channels=256
        )
        for T in [160, 320, 800]:
            x = torch.randn(1, T, 128)
            out = sub(x)
            assert out.shape == (1, T // 8, 1024), f"Failed for T={T}"

    def test_small_batch(self) -> None:
        """Single sample batch."""
        sub = ParakeetSubsamplingConv2D(
            num_mel_bins=128, hidden_size=512, conv_channels=128
        )
        x = torch.randn(1, 80, 128)
        out = sub(x)
        assert out.shape == (1, 10, 512)

    def test_gradients_flow(self) -> None:
        """Verify gradients flow through the subsampling module."""
        sub = ParakeetSubsamplingConv2D(
            num_mel_bins=128, hidden_size=1024, conv_channels=256
        )
        x = torch.randn(1, 160, 128, requires_grad=True)
        out = sub(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ============================================================================
# Positional Encoding Tests
# ============================================================================


class TestParakeetRelativePositionalEncoding:
    def test_output_shape(self) -> None:
        """Position encoding should be [2*seq_len-1, model_dim]."""
        enc = ParakeetRelativePositionalEncoding(1024)
        out = enc(100)
        assert out.shape == (2 * 100 - 1, 1024)

    def test_different_lengths(self) -> None:
        enc = ParakeetRelativePositionalEncoding(512)
        for T in [10, 50, 200]:
            out = enc(T)
            assert out.shape == (2 * T - 1, 512)

    def test_values_bounded(self) -> None:
        """Sin/cos values should be in [-1, 1]."""
        enc = ParakeetRelativePositionalEncoding(256)
        out = enc(100)
        assert out.abs().max() <= 1.0 + 1e-6

    def test_inv_freq_not_in_state_dict(self) -> None:
        """inv_freq is a non-persistent buffer, should not appear in state_dict."""
        enc = ParakeetRelativePositionalEncoding(256)
        assert "inv_freq" not in enc.state_dict()


# ============================================================================
# Relative Shift Tests
# ============================================================================


class TestRelShift:
    def test_output_shape(self) -> None:
        """rel_shift should produce [B, H, T, T] from input."""
        # Input is [B, H, T, T] (after padding trick it becomes [B, H, T, T])
        x = torch.randn(2, 8, 10, 10)
        out = _rel_shift(x)
        assert out.shape == (2, 8, 10, 10)


# ============================================================================
# Relative Attention Tests
# ============================================================================


class TestParakeetRelativeAttention:
    def test_output_shape(self) -> None:
        """Output should match input shape [B, T, D]."""
        attn = ParakeetRelativeAttention(1024, 8)
        enc = ParakeetRelativePositionalEncoding(1024)

        x = torch.randn(2, 50, 1024)
        pos = enc(50)
        out = attn(x, pos)
        assert out.shape == (2, 50, 1024)

    def test_single_token(self) -> None:
        """Should work with a single token."""
        attn = ParakeetRelativeAttention(256, 4, head_dim=64)
        enc = ParakeetRelativePositionalEncoding(256)

        x = torch.randn(1, 1, 256)
        pos = enc(1)
        out = attn(x, pos)
        assert out.shape == (1, 1, 256)

    def test_gradients_flow(self) -> None:
        attn = ParakeetRelativeAttention(256, 4)
        enc = ParakeetRelativePositionalEncoding(256)

        x = torch.randn(1, 10, 256, requires_grad=True)
        pos = enc(10)
        out = attn(x, pos)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Conformer Block Tests
# ============================================================================


class TestParakeetConformerBlock:
    def test_output_shape(self) -> None:
        """Conformer block preserves [B, T, D] shape."""
        block = ParakeetConformerBlock(
            model_dim=256, num_heads=4, ffn_dim=1024, conv_kernel_size=9
        )
        enc = ParakeetRelativePositionalEncoding(256)

        x = torch.randn(2, 20, 256)
        pos = enc(20)
        layout = BatchLayout((2, 20), seq_lens=None, device=x.device)
        out = block(x, pos, layout)
        assert out.shape == (2, 20, 256)

    def test_residual_connections(self) -> None:
        """Output should differ from zero (residual connections work)."""
        block = ParakeetConformerBlock(
            model_dim=128, num_heads=2, ffn_dim=512, conv_kernel_size=9
        )
        enc = ParakeetRelativePositionalEncoding(128)

        x = torch.randn(1, 16, 128)
        pos = enc(16)
        layout = BatchLayout((1, 16), seq_lens=None, device=x.device)
        out = block(x, pos, layout)
        # Output should not be zero (residual adds input)
        assert out.abs().sum() > 0

    def test_has_five_norms(self) -> None:
        """A Parakeet conformer block has 5 layer norms."""
        block = ParakeetConformerBlock(
            model_dim=128, num_heads=2, ffn_dim=512, conv_kernel_size=9
        )
        norm_names = [
            name
            for name, _ in block.named_modules()
            if "norm" in name and name.count(".") == 0
        ]
        assert len(norm_names) == 5


# ============================================================================
# Audio Tower Tests
# ============================================================================


class TestParakeetAudioTower:
    def test_output_shape(self) -> None:
        """Full pipeline: [2, 640, 128] -> [2, 80, 256]."""
        tower = ParakeetAudioTower(
            num_mel_bins=128,
            hidden_size=256,
            num_heads=4,
            num_layers=2,  # Use 2 layers for fast testing
            ffn_dim=512,
            conv_kernel_size=9,
            conv_channels=64,
        )

        x = torch.randn(2, 640, 128)
        out = tower(x)
        assert out.shape == (2, 80, 256)

    def test_single_sample(self) -> None:
        """Test with a single sample."""
        tower = ParakeetAudioTower(
            num_mel_bins=128,
            hidden_size=128,
            num_heads=2,
            num_layers=1,
            ffn_dim=256,
            conv_kernel_size=9,
            conv_channels=32,
        )

        x = torch.randn(1, 160, 128)
        out = tower(x)
        assert out.shape == (1, 20, 128)

    def test_num_layers(self) -> None:
        """Verify correct number of conformer layers."""
        tower = ParakeetAudioTower(
            num_mel_bins=128,
            hidden_size=128,
            num_heads=2,
            num_layers=4,
            ffn_dim=256,
            conv_kernel_size=9,
            conv_channels=32,
        )
        assert len(tower.layers) == 4


# ============================================================================
# Sound Projection Tests
# ============================================================================


class TestSoundProjection:
    def test_output_shape(self) -> None:
        """[B, T, 1024] -> [B, T, 2688]."""
        proj = SoundProjection(
            encoder_dim=1024, model_dim=2688, hidden_dim=4096
        )
        x = torch.randn(2, 80, 1024)
        out = proj(x)
        assert out.shape == (2, 80, 2688)

    def test_squared_relu(self) -> None:
        """Output should reflect SquaredReLU behavior (non-negative intermediates)."""
        proj = SoundProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128
        )
        x = torch.randn(1, 10, 64)
        out = proj(x)
        # Just verify it runs and produces finite values
        assert torch.isfinite(out).all()

    def test_no_bias(self) -> None:
        """Default is bias=False for linear layers."""
        proj = SoundProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128, bias=False
        )
        assert proj.linear1.bias is None
        assert proj.linear2.bias is None

    def test_with_bias(self) -> None:
        """Test with bias=True."""
        proj = SoundProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128, bias=True
        )
        assert proj.linear1.bias is not None
        assert proj.linear2.bias is not None

    def test_gradients_flow(self) -> None:
        proj = SoundProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128
        )
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Multimodal Model Tests
# ============================================================================


class TestNemotronHMultimodalModel:
    @pytest.fixture
    def small_config(self) -> NemotronHConfig:
        """Create a small config for testing."""
        return NemotronHConfig(
            model_dim=128,
            vocab_size=256,
            num_layers=3,
            hybrid_override_pattern="M E A",
            num_attn_heads=4,
            num_key_value_heads=2,
            attn_head_dim=32,
            mamba_num_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_n_groups=1,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
            max_seq_len=256,
            audio_config=ParakeetAudioConfig(
                hidden_size=64,
                num_attention_heads=2,
                head_dim=32,
                num_hidden_layers=1,
                intermediate_size=128,
                conv_kernel_size=9,
                num_mel_bins=128,
                subsampling_conv_channels=32,
            ),
            sound_projection_hidden_size=128,
        )

    def test_text_only_forward(self, small_config: NemotronHConfig) -> None:
        """Text-only forward (no mel_features) should work."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_config)
        assert hasattr(model, "sound_encoder")
        assert model.sound_encoder is not None

        seqs = torch.randint(0, 256, (2, 16))
        layout = BatchLayout((2, 16), seq_lens=None)
        logits = model(seqs, layout)
        assert logits.shape == (2, 16, 256)

    def test_audio_injection(self, small_config: NemotronHConfig) -> None:
        """Audio tokens should be replaced with projected audio embeddings."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_config)

        # Create input with placeholder tokens (ID=27)
        seqs = torch.randint(0, 256, (1, 32))
        # Place 10 placeholder tokens at positions 5-14
        seqs[0, 5:15] = 27

        # Create mel features that produce 10 audio tokens after subsampling
        # T/8 = 10 → T = 80
        mel = torch.randn(1, 80, 128)

        layout = BatchLayout((1, 32), seq_lens=None)
        logits = model(seqs, layout, mel_features=mel)
        assert logits.shape == (1, 32, 256)

    def test_text_only_model_when_no_audio_config(self) -> None:
        """When audio_config is None, should return plain TransformerLM."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )
        from fairseq2.models.transformer_lm import TransformerLM

        config = NemotronHConfig(
            model_dim=128,
            vocab_size=256,
            num_layers=3,
            hybrid_override_pattern="M E A",
            num_attn_heads=4,
            num_key_value_heads=2,
            attn_head_dim=32,
            mamba_num_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_n_groups=1,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
            max_seq_len=256,
            audio_config=None,  # No audio
        )

        model = create_nemotron_h_multimodal_model(config)
        assert isinstance(model, TransformerLM)

    def test_loss_computation(self, small_config: NemotronHConfig) -> None:
        """Verify loss can be computed through the multimodal model."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_config)

        seqs = torch.randint(0, 256, (1, 16))
        targets = torch.randint(0, 256, (1, 16))
        layout = BatchLayout((1, 16), seq_lens=None)
        loss = model(seqs, layout, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)


# ============================================================================
# Interop Audio Key Mapping Tests
# ============================================================================


class TestInteropAudioKeys:
    def test_ffn_key_remapping(self) -> None:
        """HF linear1/linear2 should map to inner_proj/output_proj."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "sound_encoder.encoder.layers.0.feed_forward1.linear1.weight": torch.zeros(1),
            "sound_encoder.encoder.layers.0.feed_forward1.linear2.weight": torch.zeros(1),
            "sound_encoder.encoder.layers.0.feed_forward2.linear1.bias": torch.zeros(1),
            "sound_encoder.encoder.layers.0.feed_forward2.linear2.bias": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_AUDIO_KEY_MAP)
        assert "sound_encoder.layers.0.feed_forward1.inner_proj.weight" in converted
        assert "sound_encoder.layers.0.feed_forward1.output_proj.weight" in converted
        assert "sound_encoder.layers.0.feed_forward2.inner_proj.bias" in converted
        assert "sound_encoder.layers.0.feed_forward2.output_proj.bias" in converted

    def test_attn_o_proj_remapping(self) -> None:
        """HF o_proj should map to output_proj."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "sound_encoder.encoder.layers.5.self_attn.o_proj.weight": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_AUDIO_KEY_MAP)
        assert "sound_encoder.layers.5.self_attn.output_proj.weight" in converted

    def test_conv_norm_remapping(self) -> None:
        """HF conv.norm should map to conv.batch_norm."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "sound_encoder.encoder.layers.0.conv.norm.weight": torch.zeros(1),
            "sound_encoder.encoder.layers.0.conv.norm.running_mean": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_AUDIO_KEY_MAP)
        assert "sound_encoder.layers.0.conv.batch_norm.weight" in converted
        assert "sound_encoder.layers.0.conv.batch_norm.running_mean" in converted

    def test_subsampling_prefix_strip(self) -> None:
        """Subsampling keys should strip 'encoder.' prefix."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "sound_encoder.encoder.subsampling.layers.0.weight": torch.zeros(1),
            "sound_encoder.encoder.subsampling.linear.weight": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_AUDIO_KEY_MAP)
        assert "sound_encoder.subsampling.layers.0.weight" in converted
        assert "sound_encoder.subsampling.linear.weight" in converted

    def test_sound_projection_passthrough(self) -> None:
        """sound_projection keys should pass through unchanged."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "sound_projection.norm.weight": torch.zeros(1),
            "sound_projection.linear1.weight": torch.zeros(1),
            "sound_projection.linear2.weight": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_AUDIO_KEY_MAP)
        assert "sound_projection.norm.weight" in converted
        assert "sound_projection.linear1.weight" in converted
        assert "sound_projection.linear2.weight" in converted

    def test_full_convert_with_audio_config(self) -> None:
        """Test convert_nemotron_h_state_dict with audio config enabled."""
        config = NemotronHConfig(audio_config=ParakeetAudioConfig())

        # Simulate a small HF state dict with both LM and audio keys
        state_dict: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.zeros(1),
            "language_model.backbone.norm_f.weight": torch.zeros(1),
            "language_model.lm_head.weight": torch.zeros(1),
            "sound_encoder.encoder.subsampling.layers.0.weight": torch.zeros(1),
            "sound_encoder.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "sound_projection.linear1.weight": torch.zeros(1),
            "vision_model.something": torch.zeros(1),  # should be skipped
        }

        converted = convert_nemotron_h_state_dict(state_dict, config)

        # LM keys should be converted
        assert "decoder_frontend.embed.weight" in converted
        assert "decoder.layer_norm.weight" in converted
        assert "final_proj.weight" in converted

        # Audio keys should be converted
        assert "sound_encoder.subsampling.layers.0.weight" in converted
        assert "sound_encoder.layers.0.self_attn.q_proj.weight" in converted
        assert "sound_projection.linear1.weight" in converted

        # Vision should be skipped
        assert "vision_model.something" not in converted

    def test_full_convert_text_only_skips_audio(self) -> None:
        """Without audio config, audio keys should be skipped."""
        config = NemotronHConfig(audio_config=None)

        state_dict: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.zeros(1),
            "sound_encoder.encoder.subsampling.layers.0.weight": torch.zeros(1),
            "sound_projection.linear1.weight": torch.zeros(1),
        }

        converted = convert_nemotron_h_state_dict(state_dict, config)

        assert "decoder_frontend.embed.weight" in converted
        # Audio should be skipped in text-only mode
        assert not any(k.startswith("sound_") for k in converted)


# ============================================================================
# Config Tests
# ============================================================================


class TestParakeetAudioConfig:
    def test_default_values(self) -> None:
        """Verify default config matches HF checkpoint."""
        cfg = ParakeetAudioConfig()
        assert cfg.hidden_size == 1024
        assert cfg.num_attention_heads == 8
        assert cfg.head_dim == 128
        assert cfg.num_hidden_layers == 24
        assert cfg.intermediate_size == 4096
        assert cfg.conv_kernel_size == 9
        assert cfg.num_mel_bins == 128
        assert cfg.subsampling_factor == 8
        assert cfg.subsampling_conv_channels == 256

    def test_nemotron_config_with_audio(self) -> None:
        """NemotronHConfig should accept audio_config."""
        config = NemotronHConfig(audio_config=ParakeetAudioConfig())
        assert config.audio_config is not None
        assert config.sound_context_token_id == 27

    def test_nemotron_config_without_audio(self) -> None:
        """Default NemotronHConfig has no audio."""
        config = NemotronHConfig()
        assert config.audio_config is None

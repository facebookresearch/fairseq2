# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH C-RADIO vision encoder (Phase 2).

Tests cover:
- CRADIOViTBlock forward pass and attention
- CRADIOViTEncoder full pipeline (patch embed, registers, pos embed, blocks)
- pixel_shuffle spatial downsampling
- VisionProjection shape and SquaredReLU behavior
- NemotronHMultimodalModel vision injection
- Interop vision key mapping
"""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.nemotron.config import (
    CRADIOVisionConfig,
    NemotronHConfig,
    ParakeetAudioConfig,
)
from fairseq2.models.nemotron.interop import (
    _HG_VISION_KEY_MAP,
    convert_nemotron_h_state_dict,
)
from fairseq2.models.nemotron.vision.encoder import CRADIOViTBlock, CRADIOViTEncoder
from fairseq2.models.nemotron.vision.projection import VisionProjection, pixel_shuffle
from fairseq2.nn import BatchLayout


# ============================================================================
# ViT Block Tests
# ============================================================================


class TestCRADIOViTBlock:
    def test_output_shape(self) -> None:
        """Block preserves [B, N, D] shape."""
        block = CRADIOViTBlock(hidden_size=256, num_heads=4, mlp_dim=512)
        x = torch.randn(2, 64, 256)
        out = block(x)
        assert out.shape == (2, 64, 256)

    def test_single_token(self) -> None:
        """Should work with a single token."""
        block = CRADIOViTBlock(hidden_size=128, num_heads=2, mlp_dim=256)
        x = torch.randn(1, 1, 128)
        out = block(x)
        assert out.shape == (1, 1, 128)

    def test_residual_nonzero(self) -> None:
        """Output should differ from input (residual connections work)."""
        block = CRADIOViTBlock(hidden_size=128, num_heads=2, mlp_dim=256)
        x = torch.randn(1, 16, 128)
        out = block(x)
        assert not torch.allclose(out, x)

    def test_gradients_flow(self) -> None:
        """Verify gradients flow through the block."""
        block = CRADIOViTBlock(hidden_size=128, num_heads=2, mlp_dim=256)
        x = torch.randn(1, 16, 128, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_has_two_norms(self) -> None:
        """A ViT block has 2 LayerNorms (norm1, norm2)."""
        block = CRADIOViTBlock(hidden_size=128, num_heads=2, mlp_dim=256)
        from torch.nn import LayerNorm

        norms = [m for m in block.modules() if isinstance(m, LayerNorm)]
        assert len(norms) == 2

    def test_fused_qkv_weight_shape(self) -> None:
        """Fused QKV weight should be [3*hidden_size, hidden_size]."""
        block = CRADIOViTBlock(hidden_size=256, num_heads=4, mlp_dim=512)
        assert block.attn_qkv.weight.shape == (3 * 256, 256)
        assert block.attn_qkv.bias is not None
        assert block.attn_qkv.bias.shape == (3 * 256,)


# ============================================================================
# ViT Encoder Tests
# ============================================================================


class TestCRADIOViTEncoder:
    def test_output_shape_default_image(self) -> None:
        """512×512 image → 32×32 grid → 1024 patches → [B, 1024, D]."""
        encoder = CRADIOViTEncoder(
            hidden_size=128,
            num_heads=2,
            num_layers=2,  # Use 2 layers for fast testing
            mlp_dim=256,
            patch_size=16,
            num_registers=10,
            max_grid_size=128,
            image_size=512,
        )
        x = torch.randn(1, 3, 512, 512)
        out = encoder(x)
        # 512/16 = 32 → 32*32 = 1024 patches (registers stripped)
        assert out.shape == (1, 1024, 128)

    def test_output_shape_small_image(self) -> None:
        """256×256 image → 16×16 grid → 256 patches."""
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=4,
            max_grid_size=32,
        )
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 256, 64)

    def test_registers_stripped(self) -> None:
        """Register tokens should not appear in output."""
        num_registers = 10
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=num_registers,
            max_grid_size=32,
        )
        x = torch.randn(1, 3, 64, 64)
        # 64/16 = 4 → 4*4 = 16 patches
        out = encoder(x)
        assert out.shape == (1, 16, 64)

    def test_position_embedding_interpolation(self) -> None:
        """Position embeddings should interpolate for non-max grid sizes."""
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=2,
            max_grid_size=32,
        )
        # Small image: 64×64 → 4×4 grid, needs interpolation from 32×32
        pos = encoder._interpolate_pos_embed(4, 4)
        assert pos.shape == (1, 16, 64)

    def test_position_embedding_max_grid(self) -> None:
        """At max grid size, no interpolation needed."""
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=2,
            max_grid_size=4,
        )
        pos = encoder._interpolate_pos_embed(4, 4)
        assert pos.shape == (1, 16, 64)

    def test_cls_token_shape(self) -> None:
        """Register tokens should have correct shape."""
        encoder = CRADIOViTEncoder(
            hidden_size=128,
            num_heads=2,
            num_layers=1,
            mlp_dim=256,
            patch_size=16,
            num_registers=10,
            max_grid_size=32,
        )
        assert encoder.cls_token.shape == (10, 128)

    def test_patch_embed_shape(self) -> None:
        """Patch embedding weight: [hidden_size, 3*patch_size^2]."""
        encoder = CRADIOViTEncoder(
            hidden_size=128,
            num_heads=2,
            num_layers=1,
            mlp_dim=256,
            patch_size=16,
            num_registers=2,
            max_grid_size=32,
        )
        assert encoder.patch_embed.weight.shape == (128, 768)  # 3*16*16=768

    def test_gradients_flow(self) -> None:
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=2,
            max_grid_size=32,
        )
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_norm_mean_std_in_state_dict(self) -> None:
        """norm_mean/norm_std should be in state dict (they are Parameters)."""
        encoder = CRADIOViTEncoder(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            mlp_dim=128,
            patch_size=16,
            num_registers=2,
            max_grid_size=4,
        )
        sd = encoder.state_dict()
        assert "norm_mean" in sd
        assert "norm_std" in sd


# ============================================================================
# Pixel Shuffle Tests
# ============================================================================


class TestPixelShuffle:
    def test_output_shape_default(self) -> None:
        """[B, 1024, 1280] with 32×32 grid, scale=0.5 → [B, 256, 5120]."""
        x = torch.randn(2, 1024, 1280)
        out = pixel_shuffle(x, 32, 32, 0.5)
        assert out.shape == (2, 256, 5120)

    def test_output_shape_small(self) -> None:
        """[B, 16, 64] with 4×4 grid, scale=0.5 → [B, 4, 256]."""
        x = torch.randn(1, 16, 64)
        out = pixel_shuffle(x, 4, 4, 0.5)
        assert out.shape == (1, 4, 256)

    def test_channel_expansion(self) -> None:
        """Channel dimension expands by 1/scale^2 = 4 for scale=0.5."""
        D = 128
        x = torch.randn(1, 64, D)
        out = pixel_shuffle(x, 8, 8, 0.5)
        assert out.shape[-1] == D * 4  # 128 * 4 = 512

    def test_spatial_reduction(self) -> None:
        """Token count reduces by scale^2 = 0.25 for scale=0.5."""
        x = torch.randn(1, 100, 64)
        out = pixel_shuffle(x, 10, 10, 0.5)
        assert out.shape[1] == 25  # 100 * 0.25 = 25

    def test_assertion_on_mismatch(self) -> None:
        """Should raise assertion if grid size doesn't match num_patches."""
        x = torch.randn(1, 10, 64)
        with pytest.raises(AssertionError):
            pixel_shuffle(x, 4, 4, 0.5)  # 4*4=16 != 10


# ============================================================================
# Vision Projection Tests
# ============================================================================


class TestVisionProjection:
    def test_output_shape(self) -> None:
        """[B, N, 5120] → [B, N, 2688]."""
        proj = VisionProjection(
            encoder_dim=5120, model_dim=2688, hidden_dim=20480
        )
        x = torch.randn(2, 256, 5120)
        out = proj(x)
        assert out.shape == (2, 256, 2688)

    def test_squared_relu(self) -> None:
        """Output should reflect SquaredReLU behavior (finite values)."""
        proj = VisionProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128
        )
        x = torch.randn(1, 10, 64)
        out = proj(x)
        assert torch.isfinite(out).all()

    def test_no_bias(self) -> None:
        """Default is bias=False for linear layers."""
        proj = VisionProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128, bias=False
        )
        assert proj.linear1.bias is None
        assert proj.linear2.bias is None

    def test_gradients_flow(self) -> None:
        proj = VisionProjection(
            encoder_dim=64, model_dim=32, hidden_dim=128
        )
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Multimodal Model Vision Tests
# ============================================================================


class TestNemotronHVisionMultimodal:
    @pytest.fixture
    def small_vision_config(self) -> NemotronHConfig:
        """Create a small config with vision for testing."""
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
            vision_config=CRADIOVisionConfig(
                hidden_size=64,
                num_attention_heads=2,
                head_dim=32,
                num_hidden_layers=1,
                intermediate_size=128,
                patch_size=16,
                num_registers=2,
                max_grid_size=8,
                image_size=64,
                downsample_ratio=0.5,
            ),
            vision_projection_hidden_size=512,
        )

    def test_vision_model_creation(self, small_vision_config: NemotronHConfig) -> None:
        """Creating multimodal model with vision config should work."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )
        from fairseq2.models.nemotron.model import NemotronHMultimodalModel

        model = create_nemotron_h_multimodal_model(small_vision_config)
        assert isinstance(model, NemotronHMultimodalModel)
        assert model.vision_encoder is not None
        assert model.vision_projection is not None
        assert model.sound_encoder is None  # No audio in this config

    def test_text_only_forward_with_vision_model(
        self, small_vision_config: NemotronHConfig
    ) -> None:
        """Text-only forward (no pixel_values) should work."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_vision_config)

        seqs = torch.randint(0, 256, (1, 16))
        layout = BatchLayout((1, 16), seq_lens=None)
        logits = model(seqs, layout)
        assert logits.shape == (1, 16, 256)

    def test_vision_injection(self, small_vision_config: NemotronHConfig) -> None:
        """Vision tokens should be replaced with projected vision embeddings."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_vision_config)

        # 64×64 image → 4×4 grid → 16 patches → pixel_shuffle → 4 tokens
        pixel_values = torch.randn(1, 3, 64, 64)

        # Create input with 4 image placeholder tokens (ID=18)
        seqs = torch.randint(0, 256, (1, 20))
        seqs[0, 5:9] = 18  # 4 placeholder tokens

        layout = BatchLayout((1, 20), seq_lens=None)
        logits = model(seqs, layout, pixel_values=pixel_values)
        assert logits.shape == (1, 20, 256)

    def test_vision_loss_computation(
        self, small_vision_config: NemotronHConfig
    ) -> None:
        """Loss computation through vision pipeline."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )

        model = create_nemotron_h_multimodal_model(small_vision_config)

        pixel_values = torch.randn(1, 3, 64, 64)
        seqs = torch.randint(0, 256, (1, 20))
        seqs[0, 5:9] = 18
        targets = torch.randint(0, 256, (1, 20))
        layout = BatchLayout((1, 20), seq_lens=None)

        loss = model(seqs, layout, targets, pixel_values=pixel_values)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_omni_model_creation(self) -> None:
        """Model with both vision and audio should create correctly."""
        from fairseq2.models.nemotron.factory import (
            create_nemotron_h_multimodal_model,
        )
        from fairseq2.models.nemotron.model import NemotronHMultimodalModel

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
            vision_config=CRADIOVisionConfig(
                hidden_size=64,
                num_attention_heads=2,
                head_dim=32,
                num_hidden_layers=1,
                intermediate_size=128,
                patch_size=16,
                num_registers=2,
                max_grid_size=8,
                image_size=64,
                downsample_ratio=0.5,
            ),
            vision_projection_hidden_size=512,
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

        model = create_nemotron_h_multimodal_model(config)
        assert isinstance(model, NemotronHMultimodalModel)
        assert model.vision_encoder is not None
        assert model.vision_projection is not None
        assert model.sound_encoder is not None
        assert model.sound_projection is not None


# ============================================================================
# Config Tests
# ============================================================================


class TestCRADIOVisionConfig:
    def test_default_values(self) -> None:
        """Verify default config matches HF checkpoint."""
        cfg = CRADIOVisionConfig()
        assert cfg.hidden_size == 1280
        assert cfg.num_attention_heads == 16
        assert cfg.head_dim == 80
        assert cfg.num_hidden_layers == 32
        assert cfg.intermediate_size == 5120
        assert cfg.patch_size == 16
        assert cfg.num_registers == 10
        assert cfg.max_grid_size == 128
        assert cfg.image_size == 512
        assert cfg.downsample_ratio == 0.5

    def test_nemotron_config_with_vision(self) -> None:
        """NemotronHConfig should accept vision_config."""
        config = NemotronHConfig(vision_config=CRADIOVisionConfig())
        assert config.vision_config is not None
        assert config.img_context_token_id == 18

    def test_nemotron_config_without_vision(self) -> None:
        """Default NemotronHConfig has no vision."""
        config = NemotronHConfig()
        assert config.vision_config is None


# ============================================================================
# Interop Vision Key Mapping Tests
# ============================================================================


class TestInteropVisionKeys:
    def test_vit_block_key_remapping(self) -> None:
        """HF vision_model.radio_model.model.blocks → vision_encoder.blocks."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "vision_model.radio_model.model.blocks.0.attn.qkv.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.attn.qkv.bias": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.attn.proj.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.mlp.fc1.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.mlp.fc2.bias": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.norm2.bias": torch.zeros(1),
        }

        converted = convert_state_dict(hf_keys, _HG_VISION_KEY_MAP)
        assert "vision_encoder.blocks.0.attn_qkv.weight" in converted
        assert "vision_encoder.blocks.0.attn_qkv.bias" in converted
        assert "vision_encoder.blocks.0.attn_proj.weight" in converted
        assert "vision_encoder.blocks.0.mlp_fc1.weight" in converted
        assert "vision_encoder.blocks.0.mlp_fc2.bias" in converted
        assert "vision_encoder.blocks.0.norm1.weight" in converted
        assert "vision_encoder.blocks.0.norm2.bias" in converted

    def test_patch_generator_key_remapping(self) -> None:
        """Patch generator keys should remap correctly."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "vision_model.radio_model.model.patch_generator.cls_token.token": torch.zeros(10, 1280),
            "vision_model.radio_model.model.patch_generator.embedder.weight": torch.zeros(1280, 768),
            "vision_model.radio_model.model.patch_generator.pos_embed": torch.zeros(1, 16384, 1280),
            "vision_model.radio_model.model.patch_generator.video_embedder.weight": torch.zeros(1280, 1536),
        }

        converted = convert_state_dict(hf_keys, _HG_VISION_KEY_MAP)
        assert "vision_encoder.cls_token" in converted
        assert "vision_encoder.patch_embed.weight" in converted
        assert "vision_encoder.pos_embed" in converted
        assert "vision_encoder.video_embedder.weight" in converted

    def test_input_conditioner_key_remapping(self) -> None:
        """Input conditioner buffers should remap."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "vision_model.radio_model.input_conditioner.norm_mean": torch.zeros(3, 1, 1),
            "vision_model.radio_model.input_conditioner.norm_std": torch.ones(3, 1, 1),
        }

        converted = convert_state_dict(hf_keys, _HG_VISION_KEY_MAP)
        assert "vision_encoder.norm_mean" in converted
        assert "vision_encoder.norm_std" in converted

    def test_mlp1_key_remapping(self) -> None:
        """mlp1.{0,1,3} should map to vision_projection.{norm,linear1,linear2}."""
        from fairseq2.models.utils.checkpoint import convert_state_dict

        hf_keys = {
            "mlp1.0.weight": torch.zeros(5120),
            "mlp1.1.weight": torch.zeros(20480, 5120),
            "mlp1.3.weight": torch.zeros(2688, 20480),
        }

        converted = convert_state_dict(hf_keys, _HG_VISION_KEY_MAP)
        assert "vision_projection.norm.weight" in converted
        assert "vision_projection.linear1.weight" in converted
        assert "vision_projection.linear2.weight" in converted

    def test_full_convert_with_vision_config(self) -> None:
        """Test convert_nemotron_h_state_dict with vision config enabled."""
        config = NemotronHConfig(vision_config=CRADIOVisionConfig())

        state_dict: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.zeros(1),
            "language_model.backbone.norm_f.weight": torch.zeros(1),
            "language_model.lm_head.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(1),
            "vision_model.radio_model.model.patch_generator.cls_token.token": torch.zeros(1),
            "mlp1.0.weight": torch.zeros(1),
            "mlp1.1.weight": torch.zeros(1),
            "sound_encoder.encoder.something": torch.zeros(1),  # should be skipped
            "sound_projection.something": torch.zeros(1),  # should be skipped
        }

        converted = convert_nemotron_h_state_dict(state_dict, config)

        # LM keys should be converted
        assert "decoder_frontend.embed.weight" in converted
        assert "decoder.layer_norm.weight" in converted
        assert "final_proj.weight" in converted

        # Vision keys should be converted
        assert "vision_encoder.blocks.0.norm1.weight" in converted
        assert "vision_encoder.cls_token" in converted
        assert "vision_projection.norm.weight" in converted
        assert "vision_projection.linear1.weight" in converted

        # Audio should be skipped (no audio config)
        assert not any(k.startswith("sound_") for k in converted)

    def test_full_convert_text_only_skips_vision(self) -> None:
        """Without vision config, vision keys should be skipped."""
        config = NemotronHConfig(vision_config=None)

        state_dict: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(1),
            "mlp1.0.weight": torch.zeros(1),
        }

        converted = convert_nemotron_h_state_dict(state_dict, config)

        assert "decoder_frontend.embed.weight" in converted
        # Vision should be skipped in text-only mode
        assert not any(k.startswith("vision_") for k in converted)
        assert not any(k.startswith("mlp1") for k in converted)

    def test_full_convert_omni(self) -> None:
        """Both audio and vision should be converted when both configs are set."""
        config = NemotronHConfig(
            vision_config=CRADIOVisionConfig(),
            audio_config=ParakeetAudioConfig(),
        )

        state_dict: dict[str, object] = {
            "language_model.backbone.embeddings.weight": torch.zeros(1),
            "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(1),
            "mlp1.0.weight": torch.zeros(1),
            "sound_encoder.encoder.subsampling.layers.0.weight": torch.zeros(1),
            "sound_projection.linear1.weight": torch.zeros(1),
        }

        converted = convert_nemotron_h_state_dict(state_dict, config)

        # All should be converted
        assert "decoder_frontend.embed.weight" in converted
        assert "vision_encoder.blocks.0.norm1.weight" in converted
        assert "vision_projection.norm.weight" in converted
        assert "sound_encoder.subsampling.layers.0.weight" in converted
        assert "sound_projection.linear1.weight" in converted

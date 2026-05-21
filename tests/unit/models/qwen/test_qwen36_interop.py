# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qwen 3.6 state dict interop (key mapping + RMSNorm conversion)."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.qwen.config import Qwen36Config, Qwen36MoeConfig
from fairseq2.models.qwen.interop import (
    _QWEN36_HG_KEY_MAP,
    _QWEN36_MOE_HG_KEY_MAP,
    _QWEN36_SKIP_PREFIXES,
    _QWEN36_VISION_KEY_MAP,
    convert_qwen36_moe_state_dict,
    convert_qwen36_state_dict,
)
from fairseq2.models.utils.checkpoint import convert_state_dict


class TestQwen36VisionKeyMap:
    """Verify the vision encoder key map covers all expected components."""

    def test_patch_embed_mapped(self) -> None:
        """model.visual.patch_embed.proj.* -> decoder_frontend.vision_encoder.patch_embed.proj.*"""
        sd = {"model.visual.patch_embed.proj.weight": torch.randn(1152, 3, 2, 16, 16)}
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_encoder.patch_embed.proj.weight" in converted

    def test_pos_embed_mapped(self) -> None:
        sd = {"model.visual.pos_embed.weight": torch.randn(2304, 1152)}
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_encoder.pos_embed.weight" in converted

    def test_block_attn_mapped(self) -> None:
        sd = {
            "model.visual.blocks.0.attn.qkv.weight": torch.randn(3456, 1152),
            "model.visual.blocks.0.attn.qkv.bias": torch.randn(3456),
            "model.visual.blocks.0.attn.proj.weight": torch.randn(1152, 1152),
            "model.visual.blocks.0.attn.proj.bias": torch.randn(1152),
        }
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_encoder.blocks.0.attn.qkv.weight" in converted
        assert "decoder_frontend.vision_encoder.blocks.0.attn.proj.bias" in converted

    def test_block_norm_mapped(self) -> None:
        sd = {
            "model.visual.blocks.5.norm1.weight": torch.randn(1152),
            "model.visual.blocks.5.norm2.weight": torch.randn(1152),
        }
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_encoder.blocks.5.norm1.weight" in converted
        assert "decoder_frontend.vision_encoder.blocks.5.norm2.weight" in converted

    def test_block_mlp_mapped(self) -> None:
        sd = {
            "model.visual.blocks.10.mlp.linear_fc1.weight": torch.randn(4304, 1152),
            "model.visual.blocks.10.mlp.linear_fc2.weight": torch.randn(1152, 4304),
        }
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_encoder.blocks.10.mlp.linear_fc1.weight" in converted
        assert "decoder_frontend.vision_encoder.blocks.10.mlp.linear_fc2.weight" in converted

    def test_merger_mapped(self) -> None:
        sd = {
            "model.visual.merger.norm.weight": torch.randn(1152),
            "model.visual.merger.norm.bias": torch.randn(1152),
            "model.visual.merger.linear_fc1.weight": torch.randn(4608, 4608),
            "model.visual.merger.linear_fc2.weight": torch.randn(5120, 4608),
        }
        converted = convert_state_dict(sd, _QWEN36_VISION_KEY_MAP)
        assert "decoder_frontend.vision_merger.norm.weight" in converted
        assert "decoder_frontend.vision_merger.norm.bias" in converted
        assert "decoder_frontend.vision_merger.linear_fc1.weight" in converted
        assert "decoder_frontend.vision_merger.linear_fc2.weight" in converted


class TestQwen36FullKeyMap:
    """Verify the combined VLM key map handles both vision and text keys."""

    def test_text_keys_use_language_model_prefix(self) -> None:
        """VLM text keys use model.language_model.* prefix."""
        sd = {
            "model.language_model.embed_tokens.weight": torch.randn(248064, 5120),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(5120, 5120),
            "model.language_model.norm.weight": torch.randn(5120),
        }
        converted = convert_state_dict(sd, _QWEN36_HG_KEY_MAP)
        assert "decoder_frontend.embed.weight" in converted
        assert "decoder.layers.0.self_attn.q_proj.weight" in converted
        assert "decoder.layer_norm.weight" in converted

    def test_lm_head_mapped(self) -> None:
        sd = {"lm_head.weight": torch.randn(248064, 5120)}
        converted = convert_state_dict(sd, _QWEN36_HG_KEY_MAP)
        assert "final_proj.weight" in converted

    def test_vision_and_text_combined(self) -> None:
        """Both vision and text keys are mapped in a single conversion."""
        sd = {
            "model.visual.blocks.0.attn.qkv.weight": torch.randn(3456, 1152),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(5120, 5120),
        }
        converted = convert_state_dict(sd, _QWEN36_HG_KEY_MAP)
        assert "decoder_frontend.vision_encoder.blocks.0.attn.qkv.weight" in converted
        assert "decoder.layers.0.self_attn.q_proj.weight" in converted


class TestQwen36MoeKeyMap:
    """Verify the MoE VLM key map handles expert layers."""

    def test_expert_keys_mapped(self) -> None:
        sd = {
            "model.language_model.layers.0.mlp.gate.weight": torch.randn(64, 2048),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(64, 1024, 2048),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(64, 2048, 1024),
        }
        converted = convert_state_dict(sd, _QWEN36_MOE_HG_KEY_MAP)
        assert "decoder.layers.0.ffn.gate.weight" in converted
        assert "decoder.layers.0.ffn.experts.gate_up_proj" in converted
        assert "decoder.layers.0.ffn.experts.down_proj" in converted

    def test_shared_expert_keys_mapped(self) -> None:
        sd = {
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(4096, 2048),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(4096, 2048),
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(2048, 4096),
            "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, 2048),
        }
        converted = convert_state_dict(sd, _QWEN36_MOE_HG_KEY_MAP)
        assert "decoder.layers.0.ffn.shared_expert.gate_proj.weight" in converted
        assert "decoder.layers.0.ffn.shared_expert.inner_proj.weight" in converted
        assert "decoder.layers.0.ffn.shared_expert.output_proj.weight" in converted
        assert "decoder.layers.0.ffn.shared_expert_gate.weight" in converted


class TestQwen36SkipPrefixes:
    """Verify MTP keys are skipped."""

    def test_mtp_keys_skipped(self) -> None:
        assert any("mtp" in prefix for prefix in _QWEN36_SKIP_PREFIXES)

    def test_visual_keys_not_skipped(self) -> None:
        """Vision keys should NOT be skipped in Qwen 3.6 (they were in 3.5)."""
        assert not any("visual" in prefix for prefix in _QWEN36_SKIP_PREFIXES)


class TestConvertQwen36StateDict:
    """Test the full convert_qwen36_state_dict function."""

    def test_mtp_keys_filtered(self) -> None:
        sd = {
            "mtp.layers.0.weight": torch.randn(10),
            "model.language_model.embed_tokens.weight": torch.randn(10, 8),
        }
        config = Qwen36Config()
        result = convert_qwen36_state_dict(sd, config)
        # MTP key should be filtered
        assert not any(k.startswith("mtp.") for k in result)
        # Text key should remain (converted)
        assert "decoder_frontend.embed.weight" in result

    def test_rmsnorm_plus_one_applied_to_text_only(self) -> None:
        """RMSNorm +1 conversion should only apply to text backbone norms,
        NOT to vision encoder norms."""
        sd = {
            # Text RMSNorm (should get +1)
            "model.language_model.layers.0.input_layernorm.weight": torch.zeros(8),
            # Vision LayerNorm (should NOT get +1)
            "model.visual.blocks.0.norm1.weight": torch.zeros(8),
        }
        config = Qwen36Config()
        result = convert_qwen36_state_dict(sd, config)

        # Text norm should have +1.0 applied
        text_norm = result["decoder.layers.0.self_attn_layer_norm.weight"]
        assert isinstance(text_norm, torch.Tensor)
        assert torch.allclose(text_norm, torch.ones(8))

        # Vision norm should remain as-is (zeros)
        vision_norm = result["decoder_frontend.vision_encoder.blocks.0.norm1.weight"]
        assert isinstance(vision_norm, torch.Tensor)
        assert torch.allclose(vision_norm, torch.zeros(8))

    def test_tied_embeddings(self) -> None:
        """With tied_embeddings, final_proj.weight should be copied from embed."""
        from fairseq2.models.qwen.config import Qwen35Config

        sd = {
            "model.language_model.embed_tokens.weight": torch.randn(10, 8),
        }
        text_config = Qwen35Config()
        text_config.tied_embeddings = True
        config = Qwen36Config(text_config=text_config)
        result = convert_qwen36_state_dict(sd, config)
        assert "final_proj.weight" in result
        assert "decoder_frontend.embed.weight" in result
        assert torch.equal(
            result["final_proj.weight"],  # type: ignore[arg-type]
            result["decoder_frontend.embed.weight"],  # type: ignore[arg-type]
        )


class TestConvertQwen36MoeStateDict:
    """Test the full convert_qwen36_moe_state_dict function."""

    def test_mtp_keys_filtered(self) -> None:
        sd = {
            "mtp.layers.0.weight": torch.randn(10),
            "model.language_model.embed_tokens.weight": torch.randn(10, 8),
        }
        config = Qwen36MoeConfig()
        result = convert_qwen36_moe_state_dict(sd, config)
        assert not any(k.startswith("mtp.") for k in result)

    def test_rmsnorm_plus_one_for_moe(self) -> None:
        sd = {
            "model.language_model.embed_tokens.weight": torch.randn(10, 8),
            "model.language_model.layers.0.input_layernorm.weight": torch.zeros(8),
        }
        config = Qwen36MoeConfig()
        result = convert_qwen36_moe_state_dict(sd, config)
        text_norm = result["decoder.layers.0.self_attn_layer_norm.weight"]
        assert isinstance(text_norm, torch.Tensor)
        assert torch.allclose(text_norm, torch.ones(8))

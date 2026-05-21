# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qwen 3.6 vision encoder components."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.qwen.config import Qwen36VisionConfig
from fairseq2.models.qwen.vision_encoder import (
    QwenVisionBlock,
    QwenVisionEncoder,
    QwenVisionMerger,
)


class TestQwenVisionBlock:
    def test_forward_shape(self) -> None:
        block = QwenVisionBlock(hidden_size=1152, num_heads=16, intermediate_size=4304)
        x = torch.randn(1, 64, 1152)
        out = block(x)
        assert out.shape == (1, 64, 1152)

    def test_forward_with_attention_mask(self) -> None:
        block = QwenVisionBlock(hidden_size=1152, num_heads=16, intermediate_size=4304)
        x = torch.randn(1, 16, 1152)
        mask = torch.zeros(1, 1, 16, 16)
        out = block(x, attention_mask=mask)
        assert out.shape == (1, 16, 1152)


class TestQwenVisionEncoder:
    @pytest.fixture
    def small_config(self) -> Qwen36VisionConfig:
        return Qwen36VisionConfig(
            depth=2,
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=256,
            out_hidden_size=128,
        )

    def test_construction(self, small_config: Qwen36VisionConfig) -> None:
        encoder = QwenVisionEncoder(small_config)
        assert len(encoder.blocks) == 2
        assert encoder.pos_embed.weight.shape == (256, 64)

    def test_forward_single_image(self, small_config: Qwen36VisionConfig) -> None:
        encoder = QwenVisionEncoder(small_config)
        # Simulate 4 patches (2x2 grid, 1 temporal)
        # Each patch: C * T * P * P = 3 * 2 * 16 * 16 = 1536
        num_patches = 4
        pixel_values = torch.randn(num_patches, 3 * 2 * 16 * 16)
        grid_thw = torch.tensor([[1, 2, 2]])
        out = encoder(pixel_values, grid_thw)
        assert out.shape == (4, 64)

    def test_param_count(self) -> None:
        """Verify standard config param count is reasonable."""
        config = Qwen36VisionConfig()
        with torch.device("meta"):
            encoder = QwenVisionEncoder(config)
        total = sum(p.numel() for p in encoder.parameters())
        # ~400M params for 27-block ViT
        assert 300_000_000 < total < 500_000_000


class TestQwenVisionMerger:
    def test_forward_shape(self) -> None:
        config = Qwen36VisionConfig(
            hidden_size=64, spatial_merge_size=2, out_hidden_size=128
        )
        merger = QwenVisionMerger(config)
        # 4 patches in 2x2 grid -> 1 merged patch
        hidden_states = torch.randn(4, 64)
        grid_thw = torch.tensor([[1, 2, 2]])
        out = merger(hidden_states, grid_thw)
        # spatial merge 2x2 -> 1 token per 2x2 group
        assert out.shape == (1, 128)

    def test_forward_multiple_images(self) -> None:
        config = Qwen36VisionConfig(
            hidden_size=64, spatial_merge_size=2, out_hidden_size=128
        )
        merger = QwenVisionMerger(config)
        # Image 1: 2x4 grid (8 patches -> 2 merged), Image 2: 2x2 grid (4 patches -> 1 merged)
        hidden_states = torch.randn(12, 64)
        grid_thw = torch.tensor([[1, 2, 4], [1, 2, 2]])
        out = merger(hidden_states, grid_thw)
        # Image1: 1*(2//2)*(4//2) = 2 merged, Image2: 1*(2//2)*(2//2) = 1 merged
        assert out.shape == (3, 128)

    def test_merger_norm_has_bias(self) -> None:
        """Merger's RMSNorm has bias (unlike text backbone RMSNorm)."""
        config = Qwen36VisionConfig(hidden_size=64, out_hidden_size=128)
        merger = QwenVisionMerger(config)
        assert hasattr(merger.norm, "bias")
        assert merger.norm.bias is not None

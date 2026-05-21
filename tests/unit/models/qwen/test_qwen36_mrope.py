# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qwen 3.6 Multimodal Rotary Position Encoding (M-RoPE)."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.qwen.mrope import MultimodalRotaryEncoder
from fairseq2.nn import BatchLayout


class TestMultimodalRotaryEncoder:
    @pytest.fixture
    def encoder(self) -> MultimodalRotaryEncoder:
        """Standard M-RoPE encoder with encoding_dim=64, sections=[11,11,10]."""
        return MultimodalRotaryEncoder(
            encoding_dim=64,
            max_seq_len=1024,
            theta=10_000_000.0,
            mrope_section=[11, 11, 10],
        )

    @pytest.fixture
    def batch_layout(self) -> BatchLayout:
        """Batch layout for B=1, S=8."""
        return BatchLayout.of(torch.zeros(1, 8))

    def test_construction(self, encoder: MultimodalRotaryEncoder) -> None:
        assert encoder.mrope_section == [11, 11, 10]
        assert encoder.encoding_dim == 64

    def test_invalid_section_sum(self) -> None:
        """Section pair count * 2 must equal encoding_dim."""
        with pytest.raises(ValueError, match="mrope_section"):
            MultimodalRotaryEncoder(
                encoding_dim=64,
                max_seq_len=1024,
                mrope_section=[10, 10, 10],  # 30*2 = 60 != 64
            )

    def test_text_only_delegates_to_parent(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """Without position_ids set, M-RoPE degenerates to standard 1D RoPE."""
        seqs = torch.randn(1, 8, 64)
        # No position_ids set -> delegates to parent's forward
        out = encoder(seqs, batch_layout)
        assert out.shape == (1, 8, 64)

    def test_text_only_matches_parent(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """Text-only M-RoPE (all sections same pos) should match standard RoPE."""
        from fairseq2.nn.position_encoder import ReferenceRotaryEncoder

        seqs = torch.randn(1, 8, 64)

        # Standard RoPE baseline
        ref = ReferenceRotaryEncoder(
            encoding_dim=64, max_seq_len=1024, theta=10_000_000.0
        )
        ref_out = ref(seqs, batch_layout)

        # M-RoPE without position_ids -> should be identical
        mrope_out = encoder(seqs, batch_layout)

        torch.testing.assert_close(mrope_out, ref_out)

    def test_mrope_with_uniform_positions(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """M-RoPE with all 3 sections using same sequential positions
        should match standard RoPE output."""
        from fairseq2.nn.position_encoder import ReferenceRotaryEncoder

        seqs = torch.randn(1, 8, 64)

        # Set position_ids where all 3 sections are the same: [0,1,2,...,7]
        positions = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        encoder.set_position_ids(positions)

        mrope_out = encoder(seqs, batch_layout)

        # Standard RoPE for comparison
        ref = ReferenceRotaryEncoder(
            encoding_dim=64, max_seq_len=1024, theta=10_000_000.0
        )
        ref_out = ref(seqs, batch_layout)

        torch.testing.assert_close(mrope_out, ref_out)

    def test_different_section_positions_differ(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """Different position IDs across sections should produce different output."""
        seqs = torch.randn(1, 8, 64)

        # Uniform positions (text-like)
        uniform_pos = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        encoder.set_position_ids(uniform_pos.clone())
        out_uniform = encoder(seqs, batch_layout)

        # Different positions per section (multimodal-like)
        multimodal_pos = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(1, 3, -1).clone()
        multimodal_pos[:, 1, :] = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])  # height
        multimodal_pos[:, 2, :] = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # width
        encoder.set_position_ids(multimodal_pos)
        out_multimodal = encoder(seqs, batch_layout)

        # They must differ since position IDs differ
        assert not torch.allclose(out_uniform, out_multimodal, atol=1e-5)

    def test_output_shape_preserved(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """Output shape must match input shape."""
        seqs = torch.randn(1, 8, 64)
        positions = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        encoder.set_position_ids(positions)
        out = encoder(seqs, batch_layout)
        assert out.shape == seqs.shape

    def test_batch_positions(self, encoder: MultimodalRotaryEncoder) -> None:
        """Batched M-RoPE with different position IDs per batch element."""
        B, S = 2, 4
        seqs = torch.randn(B, S, 64)
        layout = BatchLayout.of(torch.zeros(B, S))

        positions = torch.zeros(B, 3, S, dtype=torch.long)
        # Batch 0: sequential
        positions[0] = torch.arange(S).unsqueeze(0).expand(3, -1)
        # Batch 1: shuffled
        positions[1, 0] = torch.tensor([3, 2, 1, 0])
        positions[1, 1] = torch.tensor([0, 0, 1, 1])
        positions[1, 2] = torch.tensor([0, 1, 0, 1])

        encoder.set_position_ids(positions)
        out = encoder(seqs, layout)
        assert out.shape == (B, S, 64)

    def test_position_ids_cleared(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """After set_position_ids(None), encoder reverts to standard RoPE."""
        seqs = torch.randn(1, 8, 64)

        # Set multimodal positions
        positions = torch.zeros(1, 3, 8, dtype=torch.long)
        encoder.set_position_ids(positions)
        out_with_pos = encoder(seqs, batch_layout)

        # Clear
        encoder.set_position_ids(None)
        out_cleared = encoder(seqs, batch_layout)

        # With all-zero positions vs standard sequential -> they should differ
        assert not torch.allclose(out_with_pos, out_cleared, atol=1e-5)

    def test_dtype_preservation(
        self, encoder: MultimodalRotaryEncoder, batch_layout: BatchLayout
    ) -> None:
        """Output dtype should match input dtype."""
        seqs = torch.randn(1, 8, 64, dtype=torch.float16)
        positions = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        encoder.set_position_ids(positions)
        out = encoder(seqs, batch_layout)
        assert out.dtype == torch.float16


class TestQwen36ModelMRoPE:
    """Test M-RoPE position ID computation in Qwen36Model."""

    def test_text_only_positions_are_sequential(self) -> None:
        """Text-only: all 3 sections should have identical sequential positions."""
        from fairseq2.models.qwen.qwen36_model import Qwen36Model

        # We can test _compute_mrope_position_ids without a full model
        # by creating a minimal instance
        model = Qwen36Model.__new__(Qwen36Model)
        model.image_token_id = 248056
        model.mrope_section = [11, 11, 10]

        input_ids = torch.tensor([[10, 20, 30, 40, 50]])  # 5 text tokens
        position_ids = model._compute_mrope_position_ids(input_ids, None)

        assert position_ids is not None
        assert position_ids.shape == (1, 3, 5)
        # All 3 sections should be [0, 1, 2, 3, 4]
        expected = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        torch.testing.assert_close(position_ids, expected)

    def test_multimodal_positions_differ_across_sections(self) -> None:
        """With image tokens, sections should have different position sequences."""
        from fairseq2.models.qwen.qwen36_model import Qwen36Model

        model = Qwen36Model.__new__(Qwen36Model)
        model.image_token_id = 248056
        model.mrope_section = [11, 11, 10]

        # Sequence: [text, text, img, img, img, img, text]
        # Image: 1 frame, 2x2 grid -> after 2x2 merge: 1 token (but we have 4 image tokens
        # so grid is 1x4x4 -> after merge: 1*(4//2)*(4//2) = 4 merged tokens)
        image_token = 248056
        input_ids = torch.tensor([[10, 20, image_token, image_token, image_token, image_token, 50]])
        grid_thw = torch.tensor([[1, 4, 4]])  # 1 frame, 4x4 -> 2x2 merged = 4 tokens

        position_ids = model._compute_mrope_position_ids(input_ids, grid_thw)

        assert position_ids is not None
        assert position_ids.shape == (1, 3, 7)

        # Text tokens before image: sections are all [0, 1]
        assert position_ids[0, 0, 0].item() == 0
        assert position_ids[0, 0, 1].item() == 1

        # Image tokens should have different patterns across sections
        # Section 0 (temporal): t_idx for 1 frame should be 0 for all
        # Section 1 (height): h_idx varies
        # Section 2 (width): w_idx varies
        img_sec0 = position_ids[0, 0, 2:6]  # temporal section for image tokens
        img_sec1 = position_ids[0, 1, 2:6]  # height section
        img_sec2 = position_ids[0, 2, 2:6]  # width section

        # For a 1x4x4 grid with merge=2: h_merged=2, w_merged=2
        # Token 0: t=0, h=0, w=0
        # Token 1: t=0, h=0, w=1
        # Token 2: t=0, h=1, w=0
        # Token 3: t=0, h=1, w=1
        # Height and width sections should have different patterns
        assert not torch.equal(img_sec1, img_sec2) or not torch.equal(img_sec0, img_sec1)

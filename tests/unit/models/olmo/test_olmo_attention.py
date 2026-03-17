# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.models.olmo.attention import OLMOMultiheadAttention
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.nn import BatchLayout
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from tests.common import device


class TestOLMOMultiheadAttention:
    def test_init_rejects_mismatched_rope_encoder_dim(self) -> None:
        head_dim = 64
        # encoding_dim != head_dim
        rope = ReferenceRotaryEncoder(32, max_seq_len=16, device=device)
        sdpa = NaiveSDPA(IdentityBias())

        with pytest.raises(ValueError, match="rope_encoder.encoding_dim"):
            OLMOMultiheadAttention(
                model_dim=head_dim * 4,
                num_heads=4,
                sdpa=sdpa,
                rope_encoder=rope,
                device=device,
            )

    def test_extra_repr_shows_heads(self) -> None:
        sdpa = NaiveSDPA(IdentityBias())
        attn = OLMOMultiheadAttention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            device=device,
        )
        assert "num_heads=" in attn.extra_repr()

    def test_forward_produces_correct_shape(self) -> None:
        model_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        sdpa = NaiveSDPA(IdentityBias())
        attn = OLMOMultiheadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            sdpa=sdpa,
            device=device,
        )

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache: AttentionBiasCache = AttentionBiasCache()

        out = attn(seqs, layout, seqs, layout, seqs, bias_cache)
        assert out.shape == (batch_size, seq_len, model_dim)

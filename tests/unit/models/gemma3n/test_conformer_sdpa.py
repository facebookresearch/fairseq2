# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.audio.sdpa import Gemma3nConformerSDPA
from fairseq2.nn import BatchLayout


class TestGemma3nConformerSDPA:
    def test_output_shape(self) -> None:
        """Test that SDPA produces correct output shape."""
        config_hidden = 1536
        config_heads = 8

        sdpa = Gemma3nConformerSDPA(
            model_dim=config_hidden,
            num_heads=config_heads,
            max_left_rel_pos=13,
            max_right_rel_pos=0,
            chunk_size=12,
            left_context=13,
            right_context=0,
            logit_cap=50.0,
        )

        batch_size = 2
        seq_len = 24
        head_dim = config_hidden // config_heads

        q = torch.randn(batch_size, seq_len, config_heads, head_dim)
        k = torch.randn(batch_size, seq_len, config_heads, head_dim)
        v = torch.randn(batch_size, seq_len, config_heads, head_dim)

        layout = BatchLayout((batch_size, seq_len), seq_lens=[seq_len] * batch_size)

        output, _ = sdpa(q, layout, k, layout, v)

        assert output.shape == (batch_size, seq_len, config_heads, head_dim)

    def test_per_dim_scaling(self) -> None:
        """Test that per-dimension scaling parameter exists and has correct shape."""
        config_hidden = 1536
        config_heads = 8

        sdpa = Gemma3nConformerSDPA(
            model_dim=config_hidden,
            num_heads=config_heads,
            max_left_rel_pos=13,
            max_right_rel_pos=0,
            chunk_size=12,
            left_context=13,
            right_context=0,
            logit_cap=50.0,
        )

        head_dim = config_hidden // config_heads
        assert sdpa.per_dim_scale.shape == (head_dim,)

    def test_mask_forwarding(self) -> None:
        """Test that mask parameter is accepted and affects output."""
        config_hidden = 1536
        config_heads = 8

        sdpa = Gemma3nConformerSDPA(
            model_dim=config_hidden,
            num_heads=config_heads,
            max_left_rel_pos=13,
            max_right_rel_pos=0,
            chunk_size=12,
            left_context=13,
            right_context=0,
            logit_cap=50.0,
        )

        batch_size = 2
        seq_len = 24
        head_dim = config_hidden // config_heads

        q = torch.randn(batch_size, seq_len, config_heads, head_dim)
        k = torch.randn(batch_size, seq_len, config_heads, head_dim)
        v = torch.randn(batch_size, seq_len, config_heads, head_dim)

        layout = BatchLayout((batch_size, seq_len), seq_lens=[seq_len] * batch_size)

        # Without mask
        out_no_mask, _ = sdpa(q, layout, k, layout, v)

        # With mask (mask last half as invalid)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2 :] = True
        out_with_mask, _ = sdpa(q, layout, k, layout, v, mask=mask)

        assert out_with_mask.shape == out_no_mask.shape
        assert not torch.allclose(out_no_mask, out_with_mask)

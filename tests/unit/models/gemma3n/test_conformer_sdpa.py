# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.config import Gemma3nAudioConfig
from fairseq2.models.gemma3n.conformer_sdpa import Gemma3nConformerSDPA
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.nn import BatchLayout


class TestGemma3nConformerSDPA:
    def test_output_shape(self) -> None:
        """Test that SDPA produces correct output shape."""
        config = Gemma3nAudioConfig()

        sdpa = Gemma3nConformerSDPA(
            model_dim=config.hidden_size,
            num_heads=config.conf_num_attention_heads,
            max_left_rel_pos=config.conf_attention_context_left,
            max_right_rel_pos=config.conf_attention_context_right,
            chunk_size=config.conf_attention_chunk_size,
            left_context=config.conf_attention_context_left,
            right_context=config.conf_attention_context_right,
            logit_cap=config.conf_attention_logit_cap,
        )

        batch_size = 2
        seq_len = 24
        head_dim = config.hidden_size // config.conf_num_attention_heads

        q = torch.randn(batch_size, seq_len, config.conf_num_attention_heads, head_dim)
        k = torch.randn(batch_size, seq_len, config.conf_num_attention_heads, head_dim)
        v = torch.randn(batch_size, seq_len, config.conf_num_attention_heads, head_dim)

        layout = BatchLayout((batch_size, seq_len), seq_lens=[seq_len] * batch_size)
        bias_cache = AttentionBiasCache()

        output, _ = sdpa(q, layout, k, layout, v, bias_cache)

        assert output.shape == (batch_size, seq_len, config.conf_num_attention_heads, head_dim)

    def test_per_dim_scaling(self) -> None:
        """Test that per-dimension scaling parameter exists and has correct shape."""
        config = Gemma3nAudioConfig()

        sdpa = Gemma3nConformerSDPA(
            model_dim=config.hidden_size,
            num_heads=config.conf_num_attention_heads,
            max_left_rel_pos=config.conf_attention_context_left,
            max_right_rel_pos=config.conf_attention_context_right,
            chunk_size=config.conf_attention_chunk_size,
            left_context=config.conf_attention_context_left,
            right_context=config.conf_attention_context_right,
            logit_cap=config.conf_attention_logit_cap,
        )

        head_dim = config.hidden_size // config.conf_num_attention_heads
        assert sdpa.per_dim_scale.shape == (head_dim,)

    def test_chunked_local_mask(self) -> None:
        """Test that chunked local attention mask respects left/right context."""
        config = Gemma3nAudioConfig()

        sdpa = Gemma3nConformerSDPA(
            model_dim=config.hidden_size,
            num_heads=config.conf_num_attention_heads,
            max_left_rel_pos=config.conf_attention_context_left,
            max_right_rel_pos=config.conf_attention_context_right,
            chunk_size=config.conf_attention_chunk_size,
            left_context=13,
            right_context=0,
            logit_cap=config.conf_attention_logit_cap,
        )

        seq_len = 20
        mask = sdpa._create_chunked_local_mask(seq_len, seq_len, torch.device("cpu"))

        for i in range(seq_len):
            left_valid = max(0, i - 13)
            right_valid = i

            for j in range(seq_len):
                if left_valid <= j <= right_valid:
                    assert mask[i, j].item(), f"Expected True at ({i}, {j})"
                else:
                    assert not mask[i, j].item(), f"Expected False at ({i}, {j})"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from fairseq2.models.transformer import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
    NaiveSDPA,
    StandardMultiheadAttention,
    TorchSDPA,
)
from fairseq2.nn import BatchLayout
from tests.common import assert_close, device


class TestScaledDotProductAttention:
    # fmt: off
    @pytest.mark.parametrize("use_padding,use_bias,training",
        [
            (False, False, True),
            (True,  True,  True),
            (False, True,  True),
            (True,  False, True),
            (False, False, False),
            (False, True,  False),
        ],
    )
    # fmt: on
    def test_torch_sdpa(
        self, use_padding: bool, use_bias: bool, training: bool
    ) -> None:
        attn_bias: AttentionBias

        if use_bias:
            attn_bias = CausalAttentionBias()
        else:
            attn_bias = IdentityBias()

        torch_sdpa = TorchSDPA(attn_bias)
        naive_sdpa = NaiveSDPA(attn_bias)

        if training:
            torch_sdpa.eval()
            naive_sdpa.eval()

        kwargs = self._get_sdpa_args(use_padding)

        attn1, _ = torch_sdpa(**kwargs)
        attn2, _ = naive_sdpa(**kwargs)

        assert_close(attn1, attn2)

    @staticmethod
    def _get_sdpa_args(use_padding: bool) -> dict[str, Any]:
        batch_size = 2

        num_heads = 4

        source_seq_len = 3
        target_seq_len = 2

        k_size = 2
        v_size = 3

        def random_tensor(*args: int) -> Tensor:
            return torch.randn(*args, device=device)

        q = random_tensor(batch_size, target_seq_len, num_heads, k_size)
        k = random_tensor(batch_size, source_seq_len, num_heads, k_size)
        v = random_tensor(batch_size, source_seq_len, num_heads, v_size)

        target_shape = (batch_size, target_seq_len)
        source_shape = (batch_size, source_seq_len)

        if use_padding:
            q_layout = BatchLayout(target_shape, seq_lens=None, device=device)
            k_layout = BatchLayout(source_shape, seq_lens=[2, 3], device=device)
        else:
            q_layout = BatchLayout(target_shape, seq_lens=None, device=device)
            k_layout = BatchLayout(source_shape, seq_lens=None, device=device)

        bias_cache = AttentionBiasCache()

        return {
            "q": q,
            "q_layout": q_layout,
            "k": k,
            "k_layout": k_layout,
            "v": v,
            "bias_cache": bias_cache,
        }


class TestStandardMultiheadAttention:
    @pytest.mark.parametrize(
        "q_dim, k_dim",
        [
            (128, 192),  # encoder larger than decoder
            (256, 192),  # encoder smaller than decoder
            (256, 256),  # same size
            (256, None),  # same size, by default
        ],
    )
    def test_variable_sized_attention(self, q_dim: int, k_dim: int | None) -> None:
        """
        Testing that attention works when the keys and values have a different
        size than queries.  This may happen in encoder-decoder attention.
        """
        num_heads = 8

        attn_bias = IdentityBias()

        sdpa = NaiveSDPA(attn_bias)

        mha = StandardMultiheadAttention(
            q_dim, num_heads, sdpa, kv_dim=k_dim, device=device
        )

        batch_size = 3

        seq_len = 5
        key_len = 11

        if k_dim is None:
            k_dim = q_dim

        seqs = torch.randn((batch_size, seq_len, q_dim), device=device)

        seqs_layout = BatchLayout.of(seqs)

        keys = torch.randn((batch_size, key_len, k_dim), device=device)

        keys_layout = BatchLayout.of(keys)

        bias_cache = AttentionBiasCache()

        result = mha(
            seqs,
            seqs_layout,
            keys,
            keys_layout,
            values=keys,
            bias_cache=bias_cache,
        )

        assert result.shape == seqs.shape

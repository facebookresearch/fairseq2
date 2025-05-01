# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.transformer import ALiBiAttentionBias, CausalAttentionBias
from fairseq2.nn import BatchLayout
from tests.common import assert_close, device


class TestCausalAttentionBias:
    def test_materialize_works(self) -> None:
        attn_bias = CausalAttentionBias()

        q_layout = BatchLayout((1, 4), seq_lens=None, device=device)
        k_layout = BatchLayout((1, 6), seq_lens=None, device=device)

        bias = attn_bias.materialize(
            q_layout, k_layout, device=device, dtype=torch.float32
        )

        assert bias.shape == (4, 6)

        inf = -torch.inf

        expected_bias = torch.tensor(
            [
                [0.0, 0.0, 0.0, inf, inf, inf],
                [0.0, 0.0, 0.0, 0.0, inf, inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )

        assert_close(bias, expected_bias)

    def test_materialize_works_when_attn_window_len_is_specified(self) -> None:
        attn_bias = CausalAttentionBias(attn_window_len=2)

        q_layout = BatchLayout((1, 4), seq_lens=None, device=device)
        k_layout = BatchLayout((1, 6), seq_lens=None, device=device)

        bias = attn_bias.materialize(
            q_layout, k_layout, device=device, dtype=torch.float32
        )

        assert bias.shape == (4, 6)

        inf = -torch.inf

        expected_bias = torch.tensor(
            [
                [inf, 0.0, 0.0, inf, inf, inf],
                [inf, inf, 0.0, 0.0, inf, inf],
                [inf, inf, inf, 0.0, 0.0, inf],
                [inf, inf, inf, inf, 0.0, 0.0],
            ],
            device=device,
        )

        assert_close(bias, expected_bias)


class TestALiBiAttentionBias:
    def test_materialize_works(self) -> None:
        attn_bias = ALiBiAttentionBias(num_heads=4)

        q_layout = BatchLayout((1, 4), seq_lens=None, device=device)
        k_layout = BatchLayout((1, 4), seq_lens=None, device=device)

        bias = attn_bias.materialize(
            q_layout, k_layout, device=device, dtype=torch.float32
        )

        assert bias.shape == (4, 4, 4)

        inf = -torch.inf

        # fmt: off
        expected_bias = torch.tensor(
            [
                [
                    [0.00000,     inf,     inf,     inf],
                    [0.00000, 0.25000,     inf,     inf],
                    [0.00000, 0.25000, 0.50000,     inf],
                    [0.00000, 0.25000, 0.50000, 0.75000],
                ],
                [
                    [0.00000,     inf,     inf,     inf],
                    [0.00000, 0.06250,     inf,     inf],
                    [0.00000, 0.06250, 0.12500,     inf],
                    [0.00000, 0.06250, 0.12500, 0.18750],
                ],
                [
                    [0.00000,     inf,     inf,     inf],
                    [0.00000, 0.01562,     inf,     inf],
                    [0.00000, 0.01562, 0.03125,     inf],
                    [0.00000, 0.01562, 0.03125, 0.04688],
                ],
                [
                    [0.00000,     inf,     inf,     inf],
                    [0.00000, 0.00391,     inf,     inf],
                    [0.00000, 0.00391, 0.00781,     inf],
                    [0.00000, 0.00391, 0.00781, 0.01172],
                ],
            ],
            device=device,
        )
        # fmt: on

        assert_close(bias, expected_bias)

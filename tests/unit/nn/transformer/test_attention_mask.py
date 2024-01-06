# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn import IncrementalStateBag
from fairseq2.nn.transformer import ALiBiMaskFactory, CausalAttentionMaskFactory
from tests.common import assert_close, device


class TestCausalAttentionMaskFactory:
    def test_call_works(self) -> None:
        factory = CausalAttentionMaskFactory()

        q = torch.ones((6, 4, 3), device=device)
        k = torch.ones((6, 6, 3), device=device)

        mask = factory(seqs=q, keys=k)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 6)

        inf = -torch.inf

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf, inf, inf],
                [0.0, 0.0, inf, inf, inf, inf],
                [0.0, 0.0, 0.0, inf, inf, inf],
                [0.0, 0.0, 0.0, 0.0, inf, inf],
            ],
            device=device,
        )

        assert_close(m, expected_mask)

    def test_call_works_when_attn_window_len_is_specified(self) -> None:
        factory = CausalAttentionMaskFactory(attn_window_len=2)

        q = torch.ones((6, 4, 3), device=device)
        k = torch.ones((6, 6, 3), device=device)

        mask = factory(seqs=q, keys=k)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 6)

        inf = -torch.inf

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf, inf, inf],
                [0.0, 0.0, inf, inf, inf, inf],
                [inf, 0.0, 0.0, inf, inf, inf],
                [inf, inf, 0.0, 0.0, inf, inf],
            ],
            device=device,
        )

        assert_close(m, expected_mask)

    def test_call_works_when_state_bag_is_specified(self) -> None:
        state_bag = IncrementalStateBag(max_num_steps=10)

        state_bag.increment_step_nr(2)

        factory = CausalAttentionMaskFactory(attn_window_len=2)

        q = torch.ones((6, 3, 3), device=device)
        k = torch.ones((6, 6, 3), device=device)

        mask = factory(seqs=q, keys=k, training=False, state_bag=state_bag)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (3, 6)

        inf = -torch.inf

        expected_mask = torch.tensor(
            [
                [inf, 0.0, 0.0, inf, inf, inf],
                [inf, inf, 0.0, 0.0, inf, inf],
                [inf, inf, inf, 0.0, 0.0, inf],
            ],
            device=device,
        )

        assert_close(m, expected_mask)

        q = torch.ones((6, 1, 3), device=device)

        mask = factory(seqs=q, keys=k, training=False, state_bag=state_bag)

        assert mask is None


class TestALiBiAttentionMaskGenerator:
    def test_call_works(self) -> None:
        factory = ALiBiMaskFactory(num_attn_heads=4)

        q = torch.ones((32, 4, 3), device=device)
        k = torch.ones((32, 6, 3), device=device)

        mask = factory(seqs=q, keys=k)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 4, 6)

        inf = -torch.inf

        # fmt: off
        expected_mask = torch.tensor(
            [
                [
                    [0.00000,     inf,     inf,     inf, inf, inf],
                    [0.00000, 0.25000,     inf,     inf, inf, inf],
                    [0.00000, 0.25000, 0.50000,     inf, inf, inf],
                    [0.00000, 0.25000, 0.50000, 0.75000, inf, inf],
                ],
                [
                    [0.00000,     inf,     inf,     inf, inf, inf],
                    [0.00000, 0.06250,     inf,     inf, inf, inf],
                    [0.00000, 0.06250, 0.12500,     inf, inf, inf],
                    [0.00000, 0.06250, 0.12500, 0.18750, inf, inf],
                ],
                [
                    [0.00000,     inf,     inf,     inf, inf, inf],
                    [0.00000, 0.01562,     inf,     inf, inf, inf],
                    [0.00000, 0.01562, 0.03125,     inf, inf, inf],
                    [0.00000, 0.01562, 0.03125, 0.04688, inf, inf],
                ],
                [
                    [0.00000,     inf,     inf,     inf, inf, inf],
                    [0.00000, 0.00391,     inf,     inf, inf, inf],
                    [0.00000, 0.00391, 0.00781,     inf, inf, inf],
                    [0.00000, 0.00391, 0.00781, 0.01172, inf, inf],
                ],
            ],
            device=device,
        )
        # fmt: on

        assert_close(m, expected_mask)

    def test_call_works_in_incremental_decode(self) -> None:
        factory = ALiBiMaskFactory(num_attn_heads=4)

        q = torch.ones((32, 2, 3), device=device)
        k = torch.ones((32, 6, 3), device=device)

        state_bag = IncrementalStateBag(max_num_steps=10)

        state_bag.increment_step_nr(3)

        mask = factory(seqs=q, keys=k, training=False, state_bag=state_bag)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 2, 6)

        inf = -torch.inf

        # fmt: off
        expected_mask = torch.tensor(
            [
                [
                    [0.00000, 0.25000, 0.50000, 0.75000,    inf, inf],
                    [0.00000, 0.25000, 0.50000, 0.75000, 1.0000, inf],
                ],
                [
                    [0.00000, 0.06250, 0.12500, 0.18750,    inf, inf],
                    [0.00000, 0.06250, 0.12500, 0.18750, 0.2500, inf],
                ],
                [
                    [0.00000, 0.01562, 0.03125, 0.04688,    inf, inf],
                    [0.00000, 0.01562, 0.03125, 0.04688, 0.0625, inf],
                ],
                [
                    [0.00000, 0.00391, 0.00781, 0.01172,     inf, inf],
                    [0.00000, 0.00391, 0.00781, 0.01172, 0.01562, inf],
                ],
            ],
            device=device,
        )
        # fmt: on

        assert_close(m, expected_mask)

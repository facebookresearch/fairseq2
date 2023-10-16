# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn.transformer import ALiBiMaskFactory, GlobalCausalAttentionMaskFactory
from tests.common import assert_close, device


class TestGlobalCausalAttentionMaskFactory:
    def test_call_works(self) -> None:
        factory = GlobalCausalAttentionMaskFactory()

        mask = factory(torch.ones((6, 4, 3), device=device), None)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 4)

        inf = -torch.inf

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf],
                [0.0, 0.0, inf, inf],
                [0.0, 0.0, 0.0, inf],
                [0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )

        assert_close(m, expected_mask)

    def test_call_works_when_seq_len_is_less_than_or_equal_to_mask_size(self) -> None:
        factory = GlobalCausalAttentionMaskFactory()

        mask1 = factory(torch.ones((2, 4, 3), device=device), None)
        mask2 = factory(torch.ones((2, 4, 3), device=device), None)
        mask3 = factory(torch.ones((2, 3, 3), device=device), None)

        assert mask1 is not None
        assert mask2 is not None
        assert mask3 is not None

        m1 = mask1.materialize()
        m2 = mask2.materialize()
        m3 = mask3.materialize()

        assert m1.data_ptr() == m2.data_ptr()
        assert m1.data_ptr() == m3.data_ptr()

        assert m1.shape == (4, 4)
        assert m2.shape == (4, 4)
        assert m3.shape == (3, 3)


class TestALiBiAttentionMaskGenerator:
    def test_call_works(self) -> None:
        factory = ALiBiMaskFactory(num_heads=4)

        mask = factory(torch.ones((32, 4, 1), device=device), None)

        assert mask is not None

        m = mask.materialize()

        assert m.shape == (4, 4, 4)

        inf = -torch.inf

        expected_mask = torch.tensor(
            [
                [
                    [0.00000, inf, inf, inf],
                    [0.00000, 0.25000, inf, inf],
                    [0.00000, 0.25000, 0.50000, inf],
                    [0.00000, 0.25000, 0.50000, 0.75000],
                ],
                [
                    [0.00000, inf, inf, inf],
                    [0.00000, 0.06250, inf, inf],
                    [0.00000, 0.06250, 0.12500, inf],
                    [0.00000, 0.06250, 0.12500, 0.18750],
                ],
                [
                    [0.00000, inf, inf, inf],
                    [0.00000, 0.01562, inf, inf],
                    [0.00000, 0.01562, 0.03125, inf],
                    [0.00000, 0.01562, 0.03125, 0.04688],
                ],
                [
                    [0.00000, inf, inf, inf],
                    [0.00000, 0.00391, inf, inf],
                    [0.00000, 0.00391, 0.00781, inf],
                    [0.00000, 0.00391, 0.00781, 0.01172],
                ],
            ],
            device=device,
        )

        assert_close(m, expected_mask)

    def test_call_works_when_seq_len_is_less_than_or_equal_to_mask_size(self) -> None:
        num_heads = 8

        factory = ALiBiMaskFactory(num_heads)

        mask1 = factory(torch.ones((2, 4, 1), device=device), None)
        mask2 = factory(torch.ones((2, 4, 1), device=device), None)
        mask3 = factory(torch.ones((2, 3, 1), device=device), None)

        assert mask1 is not None
        assert mask2 is not None
        assert mask3 is not None

        m1 = mask1.materialize()
        m2 = mask2.materialize()
        m3 = mask3.materialize()

        assert m1.data_ptr() == m2.data_ptr()
        assert m1.data_ptr() == m3.data_ptr()

        assert m1.shape == (num_heads, 4, 4)
        assert m2.shape == (num_heads, 4, 4)
        assert m3.shape == (num_heads, 3, 3)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn.transformer import CausalAttentionMaskGenerator
from tests.common import assert_close, device


class TestCausalAttentionMaskGenerator:
    def test_call_generates_correct_mask(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask = g(torch.ones((6, 4, 3), device=device))

        assert mask.shape == (4, 4)

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

        assert_close(mask, expected_mask)

    def test_call_returns_same_mask_if_seq_len_is_equal_or_less(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask1 = g(torch.ones((2, 4, 3), device=device))
        mask2 = g(torch.ones((2, 4, 3), device=device))
        mask3 = g(torch.ones((2, 3, 3), device=device))

        assert mask1.data_ptr() == mask2.data_ptr()
        assert mask1.data_ptr() == mask3.data_ptr()

        assert mask1.shape == (4, 4)
        assert mask2.shape == (4, 4)
        assert mask3.shape == (3, 3)

    def test_call_returns_new_mask_if_seq_len_is_greater(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask1 = g(torch.ones((2, 4, 3), device=device))
        mask2 = g(torch.ones((2, 5, 3), device=device))
        mask3 = g(torch.ones((2, 8, 3), device=device))

        assert mask1.data_ptr() != mask2.data_ptr()
        assert mask1.data_ptr() != mask3.data_ptr()

        assert mask1.shape == (4, 4)
        assert mask2.shape == (5, 5)
        assert mask3.shape == (8, 8)


# class TestALiBiAttentionMaskGenerator:
#    def test_call_generates_correct_mask(self) -> None:
#        g = ALiBiAttentionMaskGenerator(num_heads=4)
#
#        mask = g(torch.ones((32, 4, 1), device=device))
#
#        assert mask.shape == (4, 4, 4)
#
#        inf = -torch.inf
#
#        expected_mask = torch.tensor(
#            [
#                [
#                    [0.00000, inf, inf, inf],
#                    [0.00000, 0.25000, inf, inf],
#                    [0.00000, 0.25000, 0.50000, inf],
#                    [0.00000, 0.25000, 0.50000, 0.75000],
#                ],
#                [
#                    [0.00000, inf, inf, inf],
#                    [0.00000, 0.06250, inf, inf],
#                    [0.00000, 0.06250, 0.12500, inf],
#                    [0.00000, 0.06250, 0.12500, 0.18750],
#                ],
#                [
#                    [0.00000, inf, inf, inf],
#                    [0.00000, 0.01562, inf, inf],
#                    [0.00000, 0.01562, 0.03125, inf],
#                    [0.00000, 0.01562, 0.03125, 0.04688],
#                ],
#                [
#                    [0.00000, inf, inf, inf],
#                    [0.00000, 0.00391, inf, inf],
#                    [0.00000, 0.00391, 0.00781, inf],
#                    [0.00000, 0.00391, 0.00781, 0.01172],
#                ],
#            ],
#            device=device,
#        )
#
#        assert_close(mask, expected_mask)
#
#        assert_close(mask, expected_mask)
#
#    def test_call_returns_same_mask_if_seq_len_is_equal_or_less(self) -> None:
#        num_heads = 8
#
#        g = ALiBiAttentionMaskGenerator(num_heads)
#
#        mask1 = g(torch.ones((2, 4, 1), device=device))
#        mask2 = g(torch.ones((2, 4, 1), device=device))
#        mask3 = g(torch.ones((2, 3, 1), device=device))
#
#        assert mask1.data_ptr() == mask2.data_ptr()
#        assert mask1.data_ptr() == mask3.data_ptr()
#
#        assert mask1.shape == (num_heads, 4, 4)
#        assert mask2.shape == (num_heads, 4, 4)
#        assert mask3.shape == (num_heads, 3, 3)
#
#    def test_call_returns_new_mask_if_seq_len_is_greater(self) -> None:
#        num_heads = 8
#
#        g = ALiBiAttentionMaskGenerator(num_heads)
#
#        mask1 = g(torch.ones((2, 4, 1), device=device))
#        mask2 = g(torch.ones((2, 5, 1), device=device))
#        mask3 = g(torch.ones((2, 8, 1), device=device))
#
#        assert mask1.data_ptr() != mask2.data_ptr()
#        assert mask1.data_ptr() != mask3.data_ptr()
#
#        assert mask1.shape == (num_heads, 4, 4)
#        assert mask2.shape == (num_heads, 5, 5)
#        assert mask3.shape == (num_heads, 8, 8)

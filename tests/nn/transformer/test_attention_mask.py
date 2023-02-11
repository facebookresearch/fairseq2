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

        mask = g(torch.ones((6, 4), device=device))

        assert mask.shape == (4, 4)

        inf = float("-inf")

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

    def test_call_generates_correct_mask_if_no_batch(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask = g(torch.ones((4,), device=device))

        assert mask.shape == (4, 4)

        inf = float("-inf")

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

        mask1 = g(torch.ones((4,), device=device))
        mask2 = g(torch.ones((4,), device=device))
        mask3 = g(torch.ones((3,), device=device))

        assert mask1.data_ptr() == mask2.data_ptr()
        assert mask1.data_ptr() == mask3.data_ptr()

        assert mask1.shape == (4, 4)
        assert mask2.shape == (4, 4)
        assert mask3.shape == (3, 3)

    def test_call_returns_new_mask_if_seq_len_is_greater(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask1 = g(torch.ones((4,), device=device))
        mask2 = g(torch.ones((5,), device=device))
        mask3 = g(torch.ones((8,), device=device))

        assert mask1.data_ptr() != mask2.data_ptr()
        assert mask1.data_ptr() != mask3.data_ptr()

        assert mask1.shape == (4, 4)
        assert mask2.shape == (5, 5)
        assert mask3.shape == (8, 8)

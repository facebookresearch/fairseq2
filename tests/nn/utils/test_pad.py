# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn.utils.pad import to_padding_mask
from tests.common import assert_equal, device


def test_to_padding_mask_with_dim1() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, max_seq_len=6)

    # fmt: off
    expected_mask = torch.tensor(
        [[False, False, False, False, True,  True],
         [False, False, True,  True,  True,  True],
         [True,  True,  True,  True,  True,  True],
         [False, False, False, False, False, True]], device=device)
    # fmt: on

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_dim2() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    seq_lens = seq_lens.unsqueeze(-1)

    mask = to_padding_mask(seq_lens, max_seq_len=6)

    # fmt: off
    expected_mask = torch.tensor(
        [[False, False, False, False, True,  True],
         [False, False, True,  True,  True,  True],
         [True,  True,  True,  True,  True,  True],
         [False, False, False, False, False, True]], device=device)
    # fmt: on

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_dim0() -> None:
    seq_lens = torch.tensor(2, device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, max_seq_len=4)

    expected_mask = torch.tensor([False, False, True, True], device=device)

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_single_seq_len() -> None:
    seq_lens = torch.tensor([4], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, max_seq_len=6)

    expected_mask = torch.tensor(
        [[False, False, False, False, True, True]], device=device
    )

    assert_equal(mask, expected_mask)

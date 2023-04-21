# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn.utils.mask import compute_mask, to_padding_mask
from tests.common import assert_equal, device


def test_compute_mask_returns_mask_with_correct_amount_of_positions_masked() -> None:
    shape = (32, 512)

    mask = compute_mask(shape, mask_len=10, mask_prob=0.65, device=device)

    assert mask is not None

    num_masked_positions = mask.sum() / mask.numel()

    # A very rudimentary confidence internal check.
    assert 0.40 <= num_masked_positions <= 0.60


def test_compute_mask_returns_mask_with_correct_amount_of_positions_masked_if_row_lens_is_specified() -> None:
    shape = (32, 512)

    row_lens = torch.full((32,), 512, device=device, dtype=torch.int64)

    mask = compute_mask(
        shape, mask_len=10, mask_prob=0.65, device=device, row_lens=row_lens
    )

    assert mask is not None

    num_masked_positions = mask.sum() / mask.numel()

    # A very rudimentary confidence internal check.
    assert 0.40 <= num_masked_positions <= 0.60


def test_compute_mask_returns_mask_with_correct_shape_device_dtype() -> None:
    shape = (4, 16)

    mask = compute_mask(shape, mask_len=4, mask_prob=1.0, device=device)

    assert mask is not None

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert mask.any()


def test_compute_mask_returns_mask_with_correct_shape_device_dtype_if_row_lens_is_specified() -> None:
    shape = (4, 16)

    row_lens = torch.tensor([16, 14, 15, 16], device="cpu")

    mask = compute_mask(
        shape, mask_len=4, mask_prob=1.0, device=device, row_lens=row_lens
    )

    assert mask is not None

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert mask.any()


def test_compute_mask_returns_none_if_rows_are_empty() -> None:
    shape = (1, 16)

    row_lens = torch.zeros((1,), device=device)

    mask = compute_mask(
        shape, mask_len=4, mask_prob=1.0, row_lens=row_lens, device=device
    )

    assert mask is None


def test_compute_mask_returns_none_if_rows_are_empty_and_min_num_masks_is_non_zero() -> None:
    shape = (1, 16)

    row_lens = torch.zeros((1,), device=device)

    mask = compute_mask(
        shape,
        mask_len=4,
        mask_prob=1.0,
        row_lens=row_lens,
        device=device,
        min_num_masks=32,
    )

    assert mask is None


def test_compute_mask_ignores_empty_rows() -> None:
    shape = (3, 16)

    row_lens = torch.tensor([16, 0, 12], device=device)

    mask = compute_mask(
        shape, mask_len=4, mask_prob=1.0, row_lens=row_lens, device=device
    )

    assert mask is not None

    # First and third columns should have masks.
    assert mask[0].any()
    assert mask[2].any()

    # Second column should not have a mask.
    assert not mask[1].any()


def test_compute_mask_ignores_rows_with_length_shorter_than_or_equal_to_mask_len() -> None:
    shape = (4, 16)

    row_lens = torch.tensor([16, 4, 5, 3], device=device)

    mask = compute_mask(
        shape, mask_len=4, mask_prob=1.0, row_lens=row_lens, device=device
    )

    assert mask is not None

    # First and third columns should have masks.
    assert mask[0].any()
    assert mask[2].any()

    # Second and fourth columns should not have a mask.
    assert not mask[1].any()
    assert not mask[3].any()


def test_to_padding_mask_with_dim1() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

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

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

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

    mask = to_padding_mask(seq_lens, mask_seq_len=4)

    expected_mask = torch.tensor([False, False, True, True], device=device)

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_single_seq_len() -> None:
    seq_lens = torch.tensor([4], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

    expected_mask = torch.tensor(
        [[False, False, False, False, True, True]], device=device
    )

    assert_equal(mask, expected_mask)

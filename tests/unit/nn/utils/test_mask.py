# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.nn.utils.mask import compute_row_mask
from tests.common import device


def test_compute_row_mask_works() -> None:
    shape = (32, 512)

    mask = compute_row_mask(shape, span_len=10, max_mask_prob=0.65, device=device)

    assert mask is not None

    num_masked = torch.count_nonzero(mask, dim=-1)

    assert num_masked[0] > 0
    assert num_masked[0] < 512

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert (num_masked == num_masked[0]).all() == True


def test_compute_row_mask_works_when_row_lens_is_specified() -> None:
    shape = (4, 16)

    row_lens = torch.tensor([16, 14, 15, 16], device="cpu")

    mask = compute_row_mask(
        shape, span_len=4, max_mask_prob=1.0, device=device, row_lens=row_lens
    )

    assert mask is not None

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert mask.any()


def test_compute_row_mask_raises_error_when_row_length_is_smaller_than_span_len() -> (
    None
):
    shape = (4, 16)

    row_lens = torch.tensor([16, 8, 5, 3], device=device)

    with pytest.raises(
        ValueError,
        match=r"^All lengths in `row_lens` must be greater than `span_len` \(4\), but at least one length is smaller\. row_lens: tensor",
    ):
        compute_row_mask(
            shape, span_len=4, max_mask_prob=1.0, row_lens=row_lens, device=device
        )

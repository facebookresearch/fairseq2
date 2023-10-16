# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.nn.padding import to_padding_mask
from tests.common import assert_equal, device


def test_to_padding_mask_works() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, 6)

    # fmt: off
    expected_mask = torch.tensor(
        [
            [True,  True,  True,  True,  False, False],
            [True,  True,  False, False, False, False],
            [False, False, False, False, False, False],
            [True,  True,  True,  True,  True,  False],
        ],
        device=device, dtype=torch.bool
    )
    # fmt: on

    assert mask is not None

    assert_equal(mask, expected_mask)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.nn import BatchLayout
from tests.common import assert_equal, device


def test_padding_mask_works() -> None:
    batch_layout = BatchLayout((4, 6), seq_lens=[4, 2, 3, 5], device=device)

    mask = batch_layout.position_indices >= 0

    # fmt: off
    expected_mask = torch.tensor(
        [
            [True, True, True,  True,  False, False],
            [True, True, False, False, False, False],
            [True, True, True,  False, False, False],
            [True, True, True,  True,  True,  False],
        ],
        device=device, dtype=torch.bool
    )
    # fmt: on

    assert mask is not None

    assert_equal(mask, expected_mask)

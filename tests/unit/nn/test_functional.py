# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch.nn.functional import cross_entropy as torch_cross_entropy

from fairseq2.nn.functional import cross_entropy
from tests.common import assert_close, device


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_cross_entropy_computes_loss_correctly(
    reduction: Literal["sum", "mean", "none"],
) -> None:
    logits = torch.randn((8, 16, 32), device=device)

    targets = torch.randint(low=0, high=32, size=(8, 16), device=device)

    loss1 = torch_cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=1, reduction=reduction
    )

    loss2 = cross_entropy(logits, targets, pad_idx=1, reduction=reduction)

    assert_close(loss1, loss2)

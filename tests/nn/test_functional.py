# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn.functional import cross_entropy, log_softmax

from fairseq2.nn.functional import nll_loss
from tests.common import assert_close, device


@pytest.mark.parametrize("reduction", ["none", "sum"])
def test_nll_loss_computes_loss_correctly(reduction: str) -> None:
    logits = torch.randn((8, 16, 32), device=device)

    targets = torch.randint(low=0, high=32, size=(8, 16), device=device)

    loss1 = cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=1, reduction=reduction
    )

    lprobs = log_softmax(logits, dim=-1)

    loss2 = nll_loss(lprobs, targets, pad_idx=1, reduction=reduction)  # type: ignore[arg-type]

    assert_close(loss1, loss2)

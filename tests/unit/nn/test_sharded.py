# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Sequential

from fairseq2.gang import FakeGang
from fairseq2.nn import ColumnShardedLinear, Linear, RowShardedLinear, get_shard_dims
from tests.common import device


def test_get_shard_dims_work() -> None:
    gang = FakeGang(device, rank=1, size=16)

    module = Sequential(
        Linear(32, 32, bias=True),
        ColumnShardedLinear(gang, 32, 32, bias=True),
        Linear(32, 32, bias=False),
        Sequential(
            RowShardedLinear(gang, 32, 32, bias=True),
        ),
    )

    dims = get_shard_dims(module)

    assert dims == {"1.weight": 0, "1.bias": 0, "3.0.weight": 1}

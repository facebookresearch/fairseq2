# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import Tensor

from fairseq2.data_type import DataType
from fairseq2.device import CPU, Device

TensorData: TypeAlias = int | float | Sequence[int] | Sequence[float]


def to_tensor(
    data: TensorData, *, dtype: DataType | None = None, device: Device | None = None
) -> Tensor:
    if device is None or device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)

    t = torch.tensor(data, dtype=dtype, device=CPU, pin_memory=True)

    return t.to(device, non_blocking=True)

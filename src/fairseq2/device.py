# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, TypeAlias

import torch

Device: TypeAlias = torch.device


class SupportsDeviceTransfer(ABC):
    @abstractmethod
    def to(self, device: Device, *, non_blocking: bool = False) -> None: ...


CPU: Final = Device("cpu")

META_DEVICE: Final = Device("meta")

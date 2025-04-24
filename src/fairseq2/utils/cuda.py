# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
from typing_extensions import override

from fairseq2.device import Device


class CudaContext(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def set_default_device(self, device: Device) -> None: ...


@final
class TorchCudaContext(CudaContext):
    @override
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    @override
    def device_count(self) -> int:
        return torch.cuda.device_count()

    @override
    def set_default_device(self, device: Device) -> None:
        torch.cuda.set_device(device)

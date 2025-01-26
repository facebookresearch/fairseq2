# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.optim import Optimizer

from fairseq2.optim.lr_scheduler._lr_scheduler import LRScheduler


class LRSchedulerHandler(ABC):
    @abstractmethod
    def create(
        self, optimizer: Optimizer, config: object, num_steps: int | None
    ) -> LRScheduler:
        ...

    @property
    @abstractmethod
    def requires_num_steps(self) -> bool:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]:
        ...


class UnknownLRSchedulerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known learning rate scheduler.")

        self.name = name


class UnspecifiedNumberOfStepsError(ValueError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(
            f"`num_steps` must be specified for the '{name}' learning rate scheduler."
        )

        self.name = name

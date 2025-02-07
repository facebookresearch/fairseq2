# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.optim import Optimizer

from fairseq2.optim._optimizer import ParameterCollection


class OptimizerHandler(ABC):
    @abstractmethod
    def create(self, params: ParameterCollection, config: object) -> Optimizer: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...

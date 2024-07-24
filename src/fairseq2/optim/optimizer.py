# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, final

import torch
from torch.optim import Optimizer


class AbstractOptimizer(ABC, Optimizer):
    """Provides a skeletal implementation of :class:`Optimizer`."""

    @final
    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = None

        prev_grad = torch.is_grad_enabled()

        try:
            torch.set_grad_enabled(self.defaults["differentiable"])

            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            self._do_step()
        finally:
            torch.set_grad_enabled(prev_grad)

        return loss

    @abstractmethod
    def _do_step(self) -> None:
        """Perform a single optimization step."""

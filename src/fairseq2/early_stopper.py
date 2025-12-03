# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, final

from typing_extensions import override


class EarlyStopper(ABC):
    """Stops training when an implementation-specific condition is not met."""

    @abstractmethod
    def should_stop(self, step_nr: int, score: float) -> bool:
        """
        Returns ``True`` if the training should be stopped; otherwise, ``False``.
        """


@final
class _NoopEarlyStopper(EarlyStopper):
    @override
    def should_stop(self, step_nr: int, score: float) -> bool:
        return False


NOOP_EARLY_STOPPER: Final = _NoopEarlyStopper()

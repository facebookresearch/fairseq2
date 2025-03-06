# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from typing_extensions import override


class EarlyStopper(ABC):
    """Stops training when an implementation-specific condition is not met."""

    @abstractmethod
    def should_stop(self, step_nr: int, score: float) -> bool:
        """
        :param step_nr: The number of the current training step.
        :para score: The validation score of the current training step.

        :returns: ``True`` if the training should be stopped; otherwise, ``False``.
        """


@final
class NoopEarlyStopper(EarlyStopper):
    @override
    def should_stop(self, step_nr: int, score: float) -> bool:
        return False

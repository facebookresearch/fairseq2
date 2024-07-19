# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol


class EarlyStopper(Protocol):
    """Stops training when an implementation-specific condition is not met."""

    def __call__(self, step_nr: int, score: float) -> bool:
        """
        :param step_nr:
            The number of the current training step.
        :para score:
            The validation score of the current training step.

        :returns:
            ``True`` if the training should be stopped; otherwise, ``False``.
        """

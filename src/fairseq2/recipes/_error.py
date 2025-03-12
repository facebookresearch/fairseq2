# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class RecipeError(Exception):
    pass


class UnitError(Exception):
    pass


class MinimumLossScaleReachedError(Exception):
    step_nr: int

    def __init__(self, step_nr: int) -> None:
        super().__init__(
            f"The gradients are scaled down to minimum at step {step_nr}. Training cannot continue."
        )

        self.step_nr = step_nr


class InconsistentGradientNormError(Exception):
    step_nr: int

    def __init__(self, step_nr: int) -> None:
        super().__init__(
            f"The gradients are inconsistent between processes at step {step_nr}. Training cannot continue."
        )

        self.step_nr = step_nr

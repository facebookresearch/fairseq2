# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import NoReturn


class InternalError(Exception):
    pass


class InvalidOperationError(Exception):
    pass


class NotSupportedError(Exception):
    pass


class StateDictError(Exception):
    @staticmethod
    def raise_if_not_empty(state_dict: dict[str, object]) -> None:
        if not state_dict:
            return

        s = ", ".join(sorted(state_dict.keys()))

        raise StateDictError(f"`state_dict` contains unexpected key(s): {s}")
class CorruptDataError(Exception):
    @staticmethod
    def raise_if_state_dict_not_empty(state_dict: dict[str, object]) -> None:
        if not state_dict:
            return

        s = ", ".join(sorted(state_dict.keys()))

        raise CorruptDataError(f"`state_dict` contains unexpected key(s) {s}.")


class FormatError(Exception):
    pass


class OperationalError(Exception):
    pass


def raise_operational_system_error(cause: OSError) -> NoReturn:
    raise OperationalError("A system error occurred.") from cause


class NumericalError(Exception):
    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr


class CorruptBatchError(Exception):
    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr


class CorruptCheckpointError(Exception):
    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr


class MinimumLossScaleReachedError(Exception):
    def __init__(self, step_nr: int, loss_scale: float, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
        self.loss_scale = loss_scale


class TrainValidationError(Exception):
    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr

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

        raise StateDictError(f"`state_dict` contains unexpected key(s) {s}.")


class OperationalError(Exception):
    pass


def raise_operational_system_error(cause: OSError) -> NoReturn:
    raise OperationalError("A system error occurred.") from cause

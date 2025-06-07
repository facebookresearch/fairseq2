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


class OperationalError(Exception):
    pass


def raise_operational_system_error(cause: OSError) -> NoReturn:
    raise OperationalError("A system error occurred.") from cause

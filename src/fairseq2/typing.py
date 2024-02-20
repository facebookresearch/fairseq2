# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Final, TypeVar

from torch import device, dtype
from typing_extensions import TypeAlias

Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

META: Final = Device("meta")


F = TypeVar("F", bound=Callable[..., Any])


def override(f: F) -> F:
    return f

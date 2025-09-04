# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import fairseq2n  # Report any fairseq2n initialization error eagerly.

import fairseq2.runtime.dependency
from fairseq2.error import InvalidOperationError
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

__version__ = "0.5.0"


_in_call: bool = False


def init_fairseq2(
    *, extras: Callable[[DependencyContainer], None] | None = None
) -> DependencyResolver:
    from fairseq2.composition import _register_library

    global _in_call

    if fairseq2.runtime.dependency._resolver is not None:
        raise InvalidOperationError("`init_fairseq2()` is already called.")

    if _in_call:
        raise InvalidOperationError("`init_fairseq2()` cannot be called recursively.")

    _in_call = True

    container = DependencyContainer()

    try:
        _register_library(container)

        if extras is not None:
            extras(container)
    finally:
        _in_call = False

    fairseq2.runtime.dependency._resolver = container

    return container

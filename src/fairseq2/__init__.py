# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.5.0.dev0"

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

from fairseq2.context import RuntimeContext
from fairseq2.error import InternalError
from fairseq2.utils.progress import ProgressReporter

_default_context: RuntimeContext | None = None

_setting_up: bool = False


def setup_fairseq2(progress_reporter: ProgressReporter | None = None) -> RuntimeContext:
    """
    Sets up fairseq2.

    As part of the initialization, this function also registers extensions
    with via setuptools' `entry-point`__ mechanism. See
    :doc:`/basics/runtime_extensions` for more information.

    .. important::

        This function must be called before using any of the fairseq2 APIs.

    .. __: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    from fairseq2.setup import setup_library

    global _default_context
    global _setting_up

    if _default_context is not None:
        return _default_context

    if _setting_up:
        raise RuntimeError("`setup_fairseq2()` cannot be called recursively.")

    _setting_up = True

    try:
        context = setup_library(progress_reporter)
    finally:
        _setting_up = False

    _default_context = context

    return context


def get_runtime_context() -> RuntimeContext:
    if _default_context is None:
        setup_fairseq2()

    if _default_context is None:
        raise InternalError("fairseq2 is not initialized.")

    return _default_context

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.5.0.dev0"

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

from fairseq2.composition import register_library
from fairseq2.dependency import DependencyResolver, StandardDependencyContainer
from fairseq2.error import InternalError

_default_resolver: DependencyResolver | None = None

_in_call: bool = False


def setup_fairseq2() -> DependencyResolver:
    """
    Sets up fairseq2.

    As part of the initialization, this function also registers extensions
    with via setuptools' `entry-point`__ mechanism. See
    :doc:`/basics/runtime_extensions` for more information.

    .. important::

        This function must be called before using any of the fairseq2 APIs.

    .. __: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """

    global _default_resolver
    global _in_call

    if _default_resolver is None:
        if _in_call:
            raise RuntimeError("`setup_fairseq2()` cannot be called recursively.")

        _in_call = True

        container = StandardDependencyContainer()

        try:
            register_library(container)
        finally:
            _in_call = False

        _default_resolver = container

    return _default_resolver


def get_dependency_resolver() -> DependencyResolver:
    if _default_resolver is None:
        setup_fairseq2()

    if _default_resolver is None:
        raise InternalError("`_default_resolver` is `None`.")

    return _default_resolver

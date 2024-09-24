# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from importlib import import_module

from importlib_metadata import entry_points

from fairseq2.dependency import (
    DependencyContainer,
    StandardDependencyContainer,
    _set_container,
)
from fairseq2.logging import get_log_writer

log = get_log_writer(__name__)


_setup_called = False


def setup_fairseq2() -> None:
    """
    Sets up fairseq2 by initializing its global :class:`DependencyContainer`.

    As part of the initialization, this function also registers external
    objects with the container via setuptools' `entry-point`__ mechanism. See
    :doc:`/topics/runtime_extensions` for more information.

    .. important::

        This function must be called before using any of the fairseq2 APIs.

    .. __: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    global _setup_called

    if _setup_called:
        return

    _setup_called = True

    container = StandardDependencyContainer()

    _set_container(container)

    _setup_library(container)
    _setup_extensions(container)

    _setup_legacy_extensions()


def setup(container: DependencyContainer) -> None:
    _setup_library(container)
    _setup_extensions(container)


def _setup_library(container: DependencyContainer) -> None:
    modules = [
        "fairseq2.device",
    ]

    for name in modules:
        module = import_module(name)

        register_objects = getattr(module, "register_objects")

        register_objects(container)


def _setup_extensions(container: DependencyContainer) -> None:
    for entry_point in entry_points(group="fairseq2.extension"):
        try:
            setup_extension = entry_point.load()

            setup_extension(container)
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 extension function."
            ) from None
        except Exception as ex:
            if "FAIRSEQ2_EXTENSION_TRACE" in os.environ:
                raise RuntimeError(
                    f"The extension function at '{entry_point.value}' has failed. See nested exception for details."
                ) from ex

            log.warning("The extension function at '{}' has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip


def _setup_legacy_extensions() -> None:
    for entry_point in entry_points(group="fairseq2"):
        try:
            setup_extension = entry_point.load()

            setup_extension()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 extension function."
            ) from None
        except Exception as ex:
            if "FAIRSEQ2_EXTENSION_TRACE" in os.environ:
                raise RuntimeError(
                    f"The extension function at '{entry_point.value}' has failed. See nested exception for details."
                ) from ex

            log.warning("The extension function at '{}' has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

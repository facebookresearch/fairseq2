# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.3.0.dev0"

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

import fairseq2.datasets
import fairseq2.models

# isort: split

import os

from importlib_metadata import entry_points

from fairseq2.assets import default_asset_store, register_assets
from fairseq2.extensions import ExtensionError
from fairseq2.logging import log

_setup_complete = False


def setup_fairseq2() -> None:
    """
    Sets up fairseq2.

    As part of the initialization, this function also registers external
    objects with via setuptools' `entry-point`__ mechanism. See
    :doc:`/topics/runtime_extensions` for more information.

    .. important::

        This function must be called before using any of the fairseq2 APIs.

    .. __: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    global _setup_complete

    if _setup_complete:
        return

    # Mark as complete early on to avoid recursive calls.
    _setup_complete = True

    register_assets(default_asset_store)

    _setup_legacy_extensions()


def setup_extensions() -> None:  # compat
    setup_fairseq2()


def _setup_legacy_extensions() -> None:
    should_trace = "FAIRSEQ2_EXTENSION_TRACE" in os.environ

    for entry_point in entry_points(group="fairseq2"):
        try:
            setup_extension = entry_point.load()

            setup_extension()
        except TypeError:
            if should_trace:
                raise ExtensionError(
                    f"The '{entry_point.value}' entry point is not a valid fairseq2 setup function."
                ) from None

            log.warning("The '{}' entry point is not a valid fairseq2 setup function. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    f"The '{entry_point.value}' fairseq2 setup function has failed. See the nested exception for details."
                ) from ex

            log.warning("The '{}' fairseq2 setup function has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

        if should_trace:
            log.info("The `{}` fairseq2 setup function run successfully.", entry_point.value)  # fmt: skip

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

from fairseq2.logging import get_log_writer

log = get_log_writer(__name__)

_setup_complete = False


def setup_fairseq2() -> None:
    setup_extensions()


# legacy
def setup_extensions() -> None:
    global _setup_complete

    if _setup_complete:
        return

    # Mark as complete early on to avoid recursive calls.
    _setup_complete = True

    for entry_point in entry_points(group="fairseq2"):
        try:
            setup_extension = entry_point.load()

            setup_extension()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 setup function."
            ) from None
        except Exception as ex:
            if "FAIRSEQ2_EXTENSION_TRACE" in os.environ:
                raise RuntimeError(
                    f"The setup function at '{entry_point.value}' has failed. See nested exception for details."
                ) from ex

            log.warning(
                "The setup function at '{}' has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.",
                entry_point.value,
            )

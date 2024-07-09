# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.3.0.dev0"

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

import fairseq2.datasets
import fairseq2.models

# isort: split

from importlib_metadata import entry_points

_setup_complete = False


def setup_extensions() -> None:
    global _setup_complete

    if _setup_complete:
        return

    # Mark as complete early on to avoid recursive calls.
    _setup_complete = True

    for entry_point in entry_points(group="fairseq2"):
        setup_extension = entry_point.load()

        try:
            setup_extension()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 setup function."
            )

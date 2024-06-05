# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

from importlib_metadata import entry_points

from fairseq2.datasets import _register_datasets
from fairseq2.models import _register_models

__version__ = "0.3.0.dev0"


_register_models()

_register_datasets()


_setup_complete = False


def setup_extensions() -> None:
    global _setup_complete

    if _setup_complete:
        return

    for entry_point in entry_points(group="fairseq2"):
        setup_extension = entry_point.load()

        try:
            setup_extension()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 setup function."
            )

    _setup_complete = True

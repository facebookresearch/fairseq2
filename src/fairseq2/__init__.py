# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib_metadata import entry_points

__version__ = "0.3.0.dev0"


# Report any fairseq2n initialization error eagerly.
import fairseq2n

# isort: split

from fairseq2.datasets import _register_datasets
from fairseq2.models import _register_models
from fairseq2.utils.yaml import _register_yaml_representers

_register_datasets()

_register_models()

_register_yaml_representers()


def setup_extensions() -> None:
    for entry_point in entry_points(group="fairseq2"):
        setup_extension = entry_point.load()

        try:
            setup_extension()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 extension setup function."
            )

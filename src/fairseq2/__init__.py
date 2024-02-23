# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module

from importlib_metadata import entry_points

# Report any fairseq2n initialization error eagerly.
import_module("fairseq2n")


# Ensure that model loaders are initialized.
import_module("fairseq2.models.llama")
import_module("fairseq2.models.mistral")
import_module("fairseq2.models.nllb")
import_module("fairseq2.models.s2t_transformer")

import_module("fairseq2.datasets.nllb")


__version__ = "0.3.0.dev0"


def setup_extensions() -> None:
    for entry_point in entry_points(group="fairseq2"):
        setup_fn = entry_point.load()

        try:
            setup_fn()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 setup function."
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib_metadata import entry_points

__version__ = "0.3.0.dev0"


# Report any fairseq2n initialization error eagerly.
import fairseq2n

# Register YAML representers.
import fairseq2.utils.yaml

# isort: split

# Register models.
import fairseq2.models.llama
import fairseq2.models.mistral
import fairseq2.models.nllb
import fairseq2.models.s2t_transformer
import fairseq2.models.w2vbert
import fairseq2.models.wav2vec2
import fairseq2.models.wav2vec2.asr

# isort: split

# Register datasets.
import fairseq2.datasets.asr
import fairseq2.datasets.instruction
import fairseq2.datasets.parallel_text


def setup_extensions() -> None:
    for entry_point in entry_points(group="fairseq2"):
        setup_fn = entry_point.load()

        try:
            setup_fn()
        except TypeError:
            raise RuntimeError(
                f"The entry point '{entry_point.value}' is not a valid fairseq2 setup function."
            )

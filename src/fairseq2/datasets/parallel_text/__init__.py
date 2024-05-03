# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.datasets.parallel_text.base import LangPair as LangPair
from fairseq2.datasets.parallel_text.base import (
    ParallelTextDataset as ParallelTextDataset,
)
from fairseq2.datasets.parallel_text.base import (
    load_parallel_text_dataset as load_parallel_text_dataset,
)

# isort: split

from fairseq2.datasets.parallel_text.nllb import _register_nllb


def _register_parallel_text_datasets() -> None:
    _register_nllb()

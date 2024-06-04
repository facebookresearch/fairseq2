# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.datasets.data_reader import DataPipelineReader as DataPipelineReader
from fairseq2.datasets.data_reader import DataReader as DataReader
from fairseq2.datasets.error import DatasetError as DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader as AbstractDatasetLoader
from fairseq2.datasets.loader import DatasetLoader as DatasetLoader
from fairseq2.datasets.loader import DelegatingDatasetLoader as DelegatingDatasetLoader

# isort: split

from fairseq2.datasets.asr import _register_asr
from fairseq2.datasets.instruction import _register_instruction
from fairseq2.datasets.parallel_text import _register_parallel_text


def _register_datasets() -> None:
    _register_asr()
    _register_instruction()
    _register_parallel_text()

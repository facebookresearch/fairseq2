# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.datasets.batching import Batching as Batching
from fairseq2.datasets.batching import LengthBatching as LengthBatching
from fairseq2.datasets.batching import StaticBatching as StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader as DataPipelineReader
from fairseq2.datasets.data_reader import DataReader as DataReader
from fairseq2.datasets.error import DatasetError as DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader as AbstractDatasetLoader
from fairseq2.datasets.loader import DatasetLoader as DatasetLoader
from fairseq2.datasets.loader import DelegatingDatasetLoader as DelegatingDatasetLoader
from fairseq2.datasets.loader import get_dataset_family as get_dataset_family
from fairseq2.datasets.loader import is_dataset_card as is_dataset_card

# isort: split

import fairseq2.datasets.asr
import fairseq2.datasets.instruction
import fairseq2.datasets.parallel_text
import fairseq2.datasets.speech
import fairseq2.datasets.text

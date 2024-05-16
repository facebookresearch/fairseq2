# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.datasets.asr.base import AsrDataset as AsrDataset
from fairseq2.datasets.asr.base import load_asr_dataset as load_asr_dataset

# isort: split

from fairseq2.datasets.asr.librispeech import _register_librispeech_asr


def _register_asr_datasets() -> None:
    _register_librispeech_asr()

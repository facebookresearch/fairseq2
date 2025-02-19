# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.datasets.asr import register_generic_asr_dataset
from fairseq2.datasets.instruction import register_generic_instruction_dataset
from fairseq2.datasets.parallel_text import register_generic_parallel_text_dataset
from fairseq2.datasets.preference import register_generic_preference_dataset
from fairseq2.datasets.speech import register_generic_speech_dataset
from fairseq2.datasets.text import register_generic_text_dataset


def register_datasets(context: RuntimeContext) -> None:
    register_generic_asr_dataset(context)
    register_generic_instruction_dataset(context)
    register_generic_parallel_text_dataset(context)
    register_generic_preference_dataset(context)
    register_generic_speech_dataset(context)
    register_generic_text_dataset(context)

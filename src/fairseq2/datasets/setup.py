# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.datasets.asr import register_asr_dataset_family
from fairseq2.datasets.instruction import register_instruction_dataset_family
from fairseq2.datasets.parallel_text import register_parallel_text_dataset_family
from fairseq2.datasets.preference import register_preference_dataset_family
from fairseq2.datasets.speech import register_speech_dataset_family
from fairseq2.datasets.text import register_text_dataset_family


def register_dataset_families(context: RuntimeContext) -> None:
    register_asr_dataset_family(context)
    register_instruction_dataset_family(context)
    register_parallel_text_dataset_family(context)
    register_preference_dataset_family(context)
    register_speech_dataset_family(context)
    register_text_dataset_family(context)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.datasets.instruction.base import InstructionDataset as InstructionDataset
from fairseq2.datasets.instruction.base import (
    load_instruction_dataset as load_instruction_dataset,
)

# isort: split

from fairseq2.datasets.instruction.generic import _register_generic


def _register_instruction_datasets() -> None:
    _register_generic()

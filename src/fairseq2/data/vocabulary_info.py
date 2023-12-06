# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


@dataclass
class VocabularyInfo:
    """Describes the vocabulary used by a tokenizer"""

    size: int
    """The size of the vocabulary."""

    unk_idx: Optional[int]
    """The index of the symbol that represents an unknown element (UNK)."""

    bos_idx: Optional[int]
    """The index of the symbol that represents the beginning of a sequence (BOS)."""

    eos_idx: Optional[int]
    """The index of the symbol that represents the end of a sequence (EOS)."""

    pad_idx: Optional[int]
    """The index of the symbol that is used to pad a sequence (PAD)."""

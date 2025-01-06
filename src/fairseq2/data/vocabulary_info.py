# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VocabularyInfo:
    """Describes the vocabulary used by a tokenizer"""

    size: int
    """The size of the vocabulary."""

    unk_idx: int | None
    """The index of the symbol that represents an unknown element (UNK)."""

    bos_idx: int | None
    """The index of the symbol that represents the beginning of a sequence (BOS)."""

    eos_idx: int | None
    """The index of the symbol that represents the end of a sequence (EOS)."""

    pad_idx: int | None
    """The index of the symbol that is used to pad a sequence (PAD)."""

    boh_idx: int | None = None
    """The index of the symbol that represents the beginning of a header (BOH)."""

    eoh_idx: int | None = None
    """The index of the symbol that represents the end of a header (EOH)."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn import BatchLayout


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    max_source_seq_len: int
    max_target_seq_len: int

    def __init__(self, max_source_seq_len: int, max_target_seq_len: int) -> None:
        """
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__()

        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len

    @abstractmethod
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
    ) -> SequenceModelOutput: ...

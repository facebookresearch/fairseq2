# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional

from overrides import override
from torch import Tensor

from fairseq2.models.sequence import SequenceBatch, SequenceModel, SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag


class DecoderModel(SequenceModel):
    """Represents a decoder model."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @override
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        return self.decode_and_project(batch.seqs, batch.seq_lens)

    @abstractmethod
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> SequenceModelOutput:
        """Decode the specified sequences and produce logits for next-step
        prediction.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        :param state_bag:
            The state bag to use for incremental evaluation.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

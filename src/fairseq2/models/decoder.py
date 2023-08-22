# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor

from fairseq2.models.sequence import SequenceBatch, SequenceModel, SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import override


class SequenceDecoder(ABC):
    """Represents a sequence decoder such as a :class:`DecoderModel`."""

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Decode the specified sequences.

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

        :returns:
            - The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is
              the batch size, :math:`S` is the target sequence length, and
              :math:`M` is the dimensionality of the model.
            - The float padding mask of the decoder output. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
              the target sequence length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
            batch size, :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param decoder_padding_mask:
            The float padding mask of the decoder output. *Shape:*
            :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
            the sequence length.
        """


class DecoderModel(SequenceModel, SequenceDecoder):
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
        decoder_output, decoder_padding_mask = self.decode(batch.seqs, batch.seq_lens)

        return self.project(decoder_output, decoder_padding_mask)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

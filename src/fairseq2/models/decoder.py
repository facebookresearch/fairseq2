# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional, Tuple

from torch import Tensor

from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceBatch, SequenceModel, SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import override


class DecoderModel(SequenceModel):
    """Represents a decoder model."""

    model_dim: int

    def __init__(self, model_dim: int, vocab_info: VocabularyInfo) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(vocab_info)

        self.model_dim = model_dim

    @override
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = self.decode(
            batch.seqs, batch.padding_mask
        )

        return self.project(decoder_output, decoder_padding_mask)

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Decode the specified sequences.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is
              the batch size, :math:`S` is the target sequence length, and
              :math:`M` is the dimensionality of the model.
            - The padding mask of the decoder output. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the target
              sequence length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
            batch size, :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param decoder_padding_mask:
            The padding mask of the decoder output. *Shape:* :math:`(N,S)`,
            where :math:`N` is the batch size and :math:`S` is the sequence
            length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from overrides import override
from torch import Tensor

from fairseq2.models.seq2seq import Seq2SeqModel, Seq2SeqModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag


class EncoderDecoderModel(Seq2SeqModel):
    """Represents an encoder-decoder model."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @override
    def forward(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
        target_seqs: Tensor,
        target_seq_lens: Optional[Tensor],
    ) -> Seq2SeqModelOutput:
        encoder_out = self.encode(source_seqs, source_seq_lens)

        return self.decode_and_project(target_seqs, target_seq_lens, encoder_out)

    @abstractmethod
    def encode(self, seqs: Tensor, seq_lens: Optional[Tensor]) -> "EncoderOutput":
        """Encode the specified source sequences.

        :param seqs:
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`,
            where :math:`N` is the batch size, :math:`S_{src}` is the source
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        """

    @abstractmethod
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_out: "EncoderOutput",
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Seq2SeqModelOutput:
        """Decode the specified target sequences and produce logits.

        :param seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`,
            where :math:`N` is the batch size, :math:`S_{tgt}` is the sequence
            length, and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        :param encoder_out:
            The encoder output to use for encoder-decoder attention.
        :param state_bag:
            The state bag to use for incremental evaluation.
        """


@dataclass
class EncoderOutput:
    """Represents the output of an encoder."""

    seqs: Tensor
    """The encoded source sequences. *Shape:* :math:`(N,S_{out},M)`, where
    :math:`N` is the batch size, :math:`S_{out}` is the output sequence length,
    and :math:`M` is the dimensionality of the model."""

    padding_mask: Optional[Tensor]
    """The float padding mask of :attr:`seqs`. *Shape:* :math:`(N,S_{out})`,
    where :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

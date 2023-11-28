# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional, Tuple

from torch import Tensor

from fairseq2.data import VocabularyInfo
from fairseq2.models.seq2seq import Seq2SeqBatch, Seq2SeqModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import override


class EncoderDecoderModel(Seq2SeqModel):
    """Represents an encoder-decoder model."""

    model_dim: int

    def __init__(self, model_dim: int, target_vocab_info: VocabularyInfo) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(target_vocab_info)

        self.model_dim = model_dim

    @override
    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        encoder_output, encoder_padding_mask = self.encode(
            batch.source_seqs, batch.source_padding_mask
        )

        decoder_output, decoder_padding_mask = self.decode(
            batch.target_seqs,
            batch.target_padding_mask,
            encoder_output,
            encoder_padding_mask,
        )

        return self.project(decoder_output, decoder_padding_mask)

    @abstractmethod
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Encode the specified source sequences.

        :param seqs:
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`,
            where :math:`N` is the batch size, :math:`S_{src}` is the source
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The encoder output. *Shape:* :math:`(N,S_{out},M)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`M` is the dimensionality of the model.
            - The padding mask of the encoder output. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Decode the specified target sequences.

        :param seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`,
            where :math:`N` is the batch size, :math:`S_{tgt}` is the target
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.
        :param encoder_padding_mask:
            The padding mask of ``encoder_output``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the encoder
            output sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where
              :math:`N` is the batch size, :math:`S_{tgt}` is the target
              sequence length, and :math:`M` is the dimensionality of the model.
            - The padding mask of the decoder output. *Shape:*
              :math:`(N,S_{tgt})`, where :math:`N` is the batch size and
              :math:`S_{tgt}` is the target sequence length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where :math:`N`
            is the batch size, :math:`S_{tgt}` is the target sequence length,
            and :math:`M` is the dimensionality of the model.
        :param decoder_padding_mask:
            The padding mask of the decoder output. *Shape:* :math:`(N,S_{tgt})`,
            where :math:`N` is the batch size and :math:`S_{tgt}` is the target
            sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

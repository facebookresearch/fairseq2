# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor
from typing_extensions import override

from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn import BatchLayout, IncrementalStateBag


class EncoderDecoderModel(Seq2SeqModel):
    """Represents an encoder-decoder model."""

    model_dim: int

    def __init__(
        self, model_dim: int, max_source_seq_len: int, max_target_seq_len: int
    ) -> None:
        """
        :param model_dim: The dimensionality of the model.
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__(max_source_seq_len, max_target_seq_len)

        self.model_dim = model_dim

    @override
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
    ) -> SequenceModelOutput:
        encoder_output, encoder_output_layout = self.encode(
            source_seqs, source_seqs_layout
        )

        decoder_output, decoder_output_layout = self.decode(
            target_seqs, target_seqs_layout, encoder_output, encoder_output_layout
        )

        return self.project(decoder_output, decoder_output_layout)

    @abstractmethod
    def encode(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        """Encode the specified source sequences.

        :param seqs:
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`,
            where :math:`N` is the batch size, :math:`S_{src}` is the source
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The encoder output. *Shape:* :math:`(N,S_{enc},M)`, where
              :math:`N` is the batch size, :math:`S_{enc}` is the encoder output
              sequence length, and :math:`M` is the dimensionality of the model.
        """

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        """Decode the specified target sequences.

        :param seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`,
            where :math:`N` is the batch size, :math:`S_{tgt}` is the target
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where
              :math:`N` is the batch size, :math:`S_{tgt}` is the target
              sequence length, and :math:`M` is the dimensionality of the model.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where :math:`N`
            is the batch size, :math:`S_{tgt}` is the target sequence length,
            and :math:`M` is the dimensionality of the model.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

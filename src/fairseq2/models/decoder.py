# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor
from typing_extensions import override

from fairseq2.models.sequence import SequenceModel, SequenceModelOutput
from fairseq2.nn import BatchLayout, IncrementalStateBag


class DecoderModel(SequenceModel):
    """Represents a decoder model."""

    model_dim: int

    def __init__(self, model_dim: int, max_seq_len: int) -> None:
        """
        :param model_dim: The dimensionality of the model.
        :param max_seq_len: The maximum length of produced sequences.
        """
        super().__init__(max_seq_len)

        self.model_dim = model_dim

    @override
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> SequenceModelOutput:
        decoder_output, decoder_output_layout = self.decode(seqs, seqs_layout)

        return self.project(decoder_output, decoder_output_layout)

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        """Decode the specified sequences.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is
              the batch size, :math:`S` is the sequence length, and :math:`M` is
              the dimensionality of the model.
            - The padding mask of the decoder output. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the sequence
              length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
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

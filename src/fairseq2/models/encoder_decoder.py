# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module

from fairseq2.nn.incremental_state import IncrementalStateBag


class EncoderDecoderModel(Module, ABC):
    """Represents an encoder-decoder model."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    def forward(
        self,
        src_seqs: Tensor,
        src_seq_lens: Optional[Tensor],
        tgt_seqs: Tensor,
        tgt_seq_lens: Optional[Tensor],
    ) -> Tensor:
        """
        :param src_seqs:
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`,
            where :math:`N` is the batch size, :math:`S_{src}` is the source
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param src_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``src_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        :param tgt_seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`,
            where :math:`N` is the batch size, :math:`S_{tgt}` is the target
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param tgt_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``tgt_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.

        :returns:
            The logits of ``tgt_seqs``. The caller should apply a softmax
            function to obtain the next-step probabilities. *Shape:*
            :math:`(N,S_{tgt},D)`, where :math:`N` is the batch size,
            :math:`S_{tgt}` is the target sequence length, and :math:`D` is the
            size of the output embedding dictionary.
        """
        enc_out, enc_padding_mask = self.encode(src_seqs, src_seq_lens)

        return self.decode_and_project(
            tgt_seqs, tgt_seq_lens, enc_out, enc_padding_mask
        )

    @abstractmethod
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified source sequences.

        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The encoded output of ``seqs``. *Shape:* :math:`(N,S_{out},M)`,
              where :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`M` is the dimensionality of the model.
            - The float padding mask of the encoded output. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    @abstractmethod
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Decode the specified sequences and apply a projection to the decoder
        outputs to produce logits.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the output sequence length, and :math:`M` is the
            dimensionality of the model.
        :param enc_padding_mask:
            The float padding mask of ``enc_out``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the output
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The logits of ``seqs``. The caller should apply a softmax function
            to obtain the next-step probabilities. *Shape:* :math:`(N,S,D)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`D` is the size of the output embedding dictionary.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class EncoderDecoderFrontend(Module, ABC):
    """Represents an encoder-decoder model front-end."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            - The processed sequences to pass to the encoder or decoder.
              *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch size,
              :math:`S` is the sequence length, and :math:`M` is the
              dimensionality of the model.
            - The float padding mask of the processed sequences. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
              the sequence length.
        """

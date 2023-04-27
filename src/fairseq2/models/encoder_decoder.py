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
            The dimensionality of the model (i.e. inputs and outputs).
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
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`, or
            :math:`(S_{src},*)` when unbatched, where :math:`N` is the batch
            size, :math:`S_{src}` is the source sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param src_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``src_seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`,
            or :math:`()` when unbatched, where :math:`N` is the batch size.
        :param tgt_seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`, or
            :math:`(S_{tgt},*)` when unbatched, where :math:`N` is the batch
            size, :math:`S_{tgt}` is the target sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param tgt_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``tgt_seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`,
            or :math:`()` when unbatched, where :math:`N` is the batch size.

        :returns:
            The scores (i.e. logits) of ``tgt_seqs``. The caller should apply a
            softmax function to obtain the next-step probabilities. *Shape:*
            :math:`(N,S_{tgt},D)`, or :math:`(S_{tgt},D)` when unbatched, where
            :math:`N` is the batch size, :math:`S_{tgt}` is the target sequence
            length, and :math:`D` is the size of the output embedding
            dictionary.
        """
        enc_out, enc_padding_mask = self.encode(src_seqs, src_seq_lens)

        return self.decode_and_score(tgt_seqs, tgt_seq_lens, enc_out, enc_padding_mask)

    @abstractmethod
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified source sequences.

        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,*)`, or :math:`(S,*)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`, or
            :math:`()` when unbatched, where :math:`N` is the batch size.

        :returns:
            - The encoded output of ``seqs``. *Shape:* :math:`(N,S,M)`, or
              :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
              :math:`S` is the sequence length, and :math:`M` is the
              dimensionality of the model.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of encoder-decoder attention. *Shape:*
              :math:`(N,S)`, or :math:`(S)` when unbatched, where :math:`N` is
              the batch size and :math:`S` is the sequence length.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend.
        """

    @abstractmethod
    def decode_and_score(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Decode the specified sequences.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, or :math:`(S,*)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`, or
            :math:`()` when unbatched, where :math:`N` is the batch size.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M)`, or :math:`(S_{src},M)` when unbatched, where
            :math:`N` is the batch size, :math:`S_{src}` is the source sequence
            length, and :math:`M` is the dimensionality of the model.
        :param enc_padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of encoder-decoder attention. *Shape:*
            :math:`(N,S_{src})`, or :math:`(S_{src})` when unbatched, where
            :math:`N` is the batch size and :math:`S_{src}` is the source
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The scores (i.e. logits) of ``seqs``. The caller should apply a
            softmax function to obtain the next-step probabilities. *Shape:*
            :math:`(N,S,D)`, or :math:`(S,D)` when unbatched, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`D` is
            the size of the output embedding dictionary.
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
            The dimensionality of the model (i.e. inputs and outputs).
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
            The sequences to process. *Shape:* :math:`(N,S,*)`, or :math:`(S,*)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, :math:`(N,1)`, or
            :math:`()` when unbatched, where :math:`N` is the batch size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            - The processed sequences to pass to the encoder or decoder.
              *Shape:* :math:`(N,S,M)`, or :math:`(S,M)` when unbatched, where
              :math:`N` is the batch size, :math:`S` is the sequence length, and
              :math:`M` is the dimensionality of the model.
            - An array where each element represents the length of the sequence
              at the same index in the first return value. *Shape:* :math:`(N)`,
              :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is
              the batch size.
        """

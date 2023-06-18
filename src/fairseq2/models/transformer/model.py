# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

import torch.nn as nn
from overrides import final as finaloverride
from torch import Tensor

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqModelOutput
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.typing import DataType, Device


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder.
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the target domain (e.g. vocabulary).
        """
        model_dim = encoder.model_dim

        super().__init__(model_dim)

        if decoder.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {model_dim} and {decoder.model_dim} instead."
            )

        if encoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {model_dim} instead."
            )

        if decoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {model_dim} instead."
            )

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.encoder_frontend(seqs, seq_lens)

        seqs, _ = self.encoder(seqs, padding_mask)

        return seqs, padding_mask

    @finaloverride
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Seq2SeqModelOutput:
        if state_bag is None:
            seqs = seqs[:, :-1]

            if seq_lens is not None:
                seq_lens = seq_lens - 1

        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        seqs, _ = self.decoder(
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

        logits = self.final_proj(seqs)

        return Seq2SeqModelOutput(logits, self.target_pad_idx)


@final
class FinalProjection(ResettableProjection):
    """Produces logits from outputs of a Transformer decoder."""

    def __init__(
        self,
        model_dim: int,
        target_vocabulary_size: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param target_vocabulary_size:
            The size of the target domain (e.g. vocabulary).
        """
        super().__init__(
            model_dim, target_vocabulary_size, bias=False, device=device, dtype=dtype
        )

    @finaloverride
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.normal_(self.weight, std=self.input_dim**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

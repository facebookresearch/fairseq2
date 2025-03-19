# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer._frontend import TransformerFrontend
from fairseq2.nn import IncrementalStateBag, Projection
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        *,
        max_target_seq_len: int,
        target_vocab_info: VocabularyInfo,
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
            The projection to apply to decoder outputs.
        :param max_target_seq_len:
            The maximum length of sequences produced by the model.
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(encoder.model_dim, max_target_seq_len, target_vocab_info)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    @override
    def encode(
        self, seqs: Tensor, padding_mask: PaddingMask | None
    ) -> tuple[Tensor, PaddingMask | None]:
        seqs, padding_mask = self.encoder_frontend(seqs, padding_mask)

        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @override
    def decode(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        encoder_output: Tensor,
        encoder_padding_mask: PaddingMask | None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, PaddingMask | None]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    @override
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: PaddingMask | None
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info.pad_idx)

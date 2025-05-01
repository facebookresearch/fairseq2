# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection

# isort: split

from fairseq2.models.transformer._decoder import TransformerDecoder
from fairseq2.models.transformer._encoder import TransformerEncoder
from fairseq2.models.transformer._frontend import TransformerFrontend


@final
class TransformerModel(EncoderDecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    pad_idx: int | None

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        *,
        pad_idx: int | None,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ) -> None:
        """
        :param encoder_frontend: The encoder frontend.
        :param encoder: The encoder.
        :param decoder_frontend: The decoder frontend.
        :param decoder: The decoder.
        :param final_proj: The projection to apply to decoder outputs.
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__(encoder.model_dim, max_source_seq_len, max_target_seq_len)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.pad_idx = pad_idx

    @override
    def encode(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.encoder_frontend(seqs, seqs_layout)

        seqs = self.encoder(seqs, seqs_layout)

        return seqs, seqs_layout

    @override
    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        seqs = self.decoder(
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

        return seqs, seqs_layout

    @override
    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)

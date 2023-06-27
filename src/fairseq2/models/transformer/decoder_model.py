# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

from overrides import override as finaloverride
from torch import Tensor

from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import TransformerDecoder


@final
class TransformerDecoderModel(DecoderModel):
    """Represents a Transformer-based decoder model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the target domain (e.g. vocabulary).
        """
        model_dim = decoder.model_dim

        super().__init__(model_dim)

        if decoder_frontend.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {model_dim} instead."
            )

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx

    @finaloverride
    def decode_and_project(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> SequenceModelOutput:
        if state_bag is None:
            seqs = seqs[:, :-1]

            if seq_lens is not None:
                seq_lens = seq_lens - 1

        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        seqs, _ = self.decoder(seqs, padding_mask, state_bag=state_bag)

        logits = self.final_proj(seqs)

        return SequenceModelOutput(logits, self.target_pad_idx)

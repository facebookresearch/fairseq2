# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.nn.ops import CrossEntropy, cross_entropy

# isort: split

from fairseq2.models.transformer_lm._decoder import TransformerLMDecoder


@final
class TransformerLanguageModel(DecoderModel):
    """Represents a Transformer-based language model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerLMDecoder
    final_proj: Projection
    pad_idx: int | None
    loss_fn: CrossEntropy

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerLMDecoder,
        final_proj: Projection,
        *,
        pad_idx: int | None,
        max_seq_len: int,
    ) -> None:
        """
        :param decoder_frontend: The decoder frontend.
        :param decoder: The decoder.
        :param final_proj: The projection to apply to decoder outputs.
        :param max_seq_len: The maximum length of produced sequences.
        """
        super().__init__(decoder.model_dim, max_seq_len)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.pad_idx = pad_idx

        self.loss_fn = cross_entropy

    def compile_loss_function(self, *args: Any, **kwargs: Any) -> None:
        self.loss_fn = torch.compile(self.loss_fn, **kwargs)

    @override
    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        seqs = self.decoder(seqs, seqs_layout, state_bag=state_bag)

        return seqs, seqs_layout

    @override
    def project(
        self, decoder_output: Tensor, decoder_output_layout: BatchLayout
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx, loss_fn=self.loss_fn)

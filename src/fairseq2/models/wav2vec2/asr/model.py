# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal, final, overload

import torch
from torch import Tensor
from torch.nn import Dropout
from torch.nn.functional import ctc_loss, log_softmax
from typing_extensions import override

from fairseq2.models.asr import AsrModel
from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend, Wav2Vec2Masker
from fairseq2.nn import BatchLayout, Projection


@final
class Wav2Vec2AsrModel(AsrModel):
    """Represents a wav2vec 2.0 ASR model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        model_dim: int,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        final_proj: Projection,
        *,
        masker: Wav2Vec2Masker | None = None,
        final_dropout_p: float = 0.0,
    ) -> None:
        """
        :param encoder_frontend: The encoder frontend.
        :param encoder: The encoder (i.e. resolver network).
        :param masker: The feature masker.
        :param final_dropout_p: The dropout probability on resolver network
            outputs.
        """
        super().__init__()

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend

        self.encoder = encoder

        self.masker: Wav2Vec2Masker | None

        self.register_module("masker", masker)

        if final_dropout_p > 0.0:
            final_dropout = Dropout(final_dropout_p)
        else:
            final_dropout = None

        self.final_dropout: Dropout | None

        self.register_module("final_dropout", final_dropout)

        self.final_proj = final_proj

    @override
    @overload
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor, BatchLayout]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        targets_layout: BatchLayout,
        *,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor, BatchLayout]: ...

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        targets_layout: BatchLayout | None = None,
        *,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, BatchLayout] | tuple[Tensor, Tensor, BatchLayout]:
        seqs, seqs_layout, _ = self.encoder_frontend.extract_features(seqs, seqs_layout)

        seqs, _ = self.encoder_frontend.process_features(
            seqs, seqs_layout, self.masker if self.training else None
        )

        seqs = self.encoder(seqs, seqs_layout)

        if self.final_dropout is not None:
            seqs = self.final_dropout(seqs)

        logits = self.final_proj(seqs)

        if targets is None:
            return logits, seqs_layout

        if targets_layout is None:
            raise ValueError(
                "`targets_layout` must be specified when `targets` is specified."
            )

        if targets_layout.packed:
            raise ValueError("`targets` must not be a packed batch.")

        # For numerical stability run in single precision.
        # (N, S, T)
        log_probs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        log_probs_t = log_probs.transpose(0, 1)

        # ()
        loss = ctc_loss(
            log_probs=log_probs_t,
            input_lengths=seqs_layout.seq_lens_pt,
            targets=targets,
            target_lengths=targets_layout.seq_lens_pt,
            reduction="sum",
            zero_infinity=True,
        )

        if return_logits:
            return loss, logits, seqs_layout

        return loss

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"

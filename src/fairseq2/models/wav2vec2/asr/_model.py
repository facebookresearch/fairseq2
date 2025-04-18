# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.nn import Dropout
from typing_extensions import override

from fairseq2.models.asr import AsrModel, AsrModelOutput
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend, Wav2Vec2Masker
from fairseq2.nn import Projection
from fairseq2.typing import DataType, Device


@final
class Wav2Vec2AsrModel(AsrModel):
    """Represents a wav2vec 2.0 ASR model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    masker: Wav2Vec2Masker | None
    final_dropout: Dropout | None
    final_proj: Projection

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        final_proj: Projection,
        *,
        masker: Wav2Vec2Masker | None = None,
        final_dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend: The encoder frontend.
        :param encoder: The encoder (i.e. context network).
        :param masker: The feature masker.
        :param final_dropout_p: The dropout probability on context network
            outputs.
        """
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.register_module("masker", masker)

        if final_dropout_p > 0.0:
            self.final_dropout = Dropout(final_dropout_p)
        else:
            self.register_module("final_dropout", None)

        self.final_proj = final_proj

    @override
    def forward(self, batch: SequenceBatch) -> AsrModelOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask, _ = self.encoder_frontend.extract_features(
            batch.seqs, batch.padding_mask
        )

        seqs, padding_mask, _ = self.encoder_frontend.process_features(
            seqs, padding_mask, self.masker if self.training else None
        )

        seqs, padding_mask = self.encoder(seqs, padding_mask)

        if self.final_dropout is not None:
            seqs = self.final_dropout(seqs)

        logits = self.final_proj(seqs)

        return AsrModelOutput(logits, padding_mask)

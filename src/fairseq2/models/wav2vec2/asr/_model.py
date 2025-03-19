# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch.nn as nn
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.models.asr import AsrModel, AsrModelOutput
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend, Wav2Vec2Masker
from fairseq2.nn import Linear
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device


@final
class Wav2Vec2AsrModel(AsrModel):
    """Represents a wav2vec 2.0 ASR model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    masker: Wav2Vec2Masker | None
    final_dropout: Dropout | None
    final_proj: Linear
    target_vocab_info: VocabularyInfo

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        target_vocab_info: VocabularyInfo,
        *,
        masker: Wav2Vec2Masker | None = None,
        final_dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        :param masker:
            The feature masker.
        :param final_dropout_p:
            The dropout probability on context network outputs.
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

        self.final_proj = Linear(
            self.model_dim,
            target_vocab_info.size,
            bias=True,
            init_fn=_init_final_projection,
            device=device,
            dtype=dtype,
        )

        self.target_vocab_info = target_vocab_info

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


def _init_final_projection(proj: Linear) -> None:
    """Initialize ``proj`` as the final projection of a wav2vec 2.0 ASR model."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)

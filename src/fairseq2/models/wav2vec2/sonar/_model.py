# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, Optional, Tuple

from fairseq2.models.seq2seq import SonarSpeechSeq2SeqBatch

from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerEncoder, TransformerFrontend

from fairseq2.models.wav2vec2.sonar._pooler import EncoderOutputPooler
from fairseq2.nn import LayerNorm
from fairseq2.nn.padding import PaddingMask
from torch import Tensor
from torch.nn import Dropout, Module


@dataclass
class SonarSpeechEncoderOutput:
    """Dataclass for both speech SONAR encoder outputs"""

    text_embeddings: Tensor
    """ Pooled representation, derived from encoded_seqs by pooling in dim=1
    *Shape:* :math:`(N,M)`, where :math:`N` is the batch size, and :math:`M` is the
    dimensionality of the model.
    """

    speech_embeddings: Tensor
    """ Pooled representation, derived from encoded_seqs by pooling in dim=1
    *Shape:* :math:`(N,M)`, where :math:`N` is the batch size, and :math:`M` is the
    dimensionality of the model.
    """

    padding_mask: Optional[PaddingMask]
    """Optional, the floating padding mask over sequences (-inf means masked element)
    *Shape:* :math:`(N,S)`, where :math:`N` is the batch size,
    :math:`S` is the sequence length.
    """


class SonarEncoderModel(ABC, Module):
    """Abstract class for both speech and text SONAR encoder models"""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """

        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SonarSpeechEncoderOutput:
        """
        :param batch:
            The batch of sequences to process.
        :returns:
            SonarEncoderOutput
        """


@final
class SonarSpeechEncoderModel(SonarEncoderModel):

    encoder_frontend: TransformerFrontend
    masker: Wav2Vec2Masker | None
    encoder: TransformerEncoder
    layer_norm: Optional[LayerNorm]
    final_dropout: Dropout
    encoder_pooler: EncoderOutputPooler

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        layer_norm: Optional[LayerNorm],
        final_dropout_p: float,
        encoder_pooler: EncoderOutputPooler,
        masker: Wav2Vec2Masker | None = None,
    ) -> None:
        """
        :param encoder_frontend:
            The wav2vec2 encoder frontend.
        :param encoder:
            The wav2vec2 encoder model.
        :param layer_norm:
            Optional layer norm applied after wav2vec2 encoder.
        :param final_dropout_p:
            Dropout probability applied at the end of wav2vec2 encoder
        :param encoder_pooler:
            Encoder output pooler.
        """
        super().__init__(encoder.model_dim)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.final_dropout = Dropout(final_dropout_p)
        self.layer_norm = layer_norm
        self.encoder_pooler = encoder_pooler
        self.register_module("masker", masker)

    def forward(
        self, batch: SequenceBatch | SonarSpeechSeq2SeqBatch
    ) -> SonarSpeechEncoderOutput:
        if isinstance(batch, SonarSpeechSeq2SeqBatch):
            target_embeddings = batch.target_embedds
            batch = SequenceBatch(
                seqs=batch.source_seqs,
                padding_mask=batch.source_padding_mask,
                example=batch.example,
            )
        else:
            target_embeddings = None

        seqs, padding_mask, _ = self.encoder_frontend.extract_features(
            batch.seqs, batch.padding_mask
        )

        encoder_output, encoder_padding_mask, _ = (
            self.encoder_frontend.process_features(
                seqs, padding_mask, self.masker if self.training else None
            )
        )
        encoder_output, encoder_padding_mask = self.encoder(
            encoder_output, encoder_padding_mask
        )

        # This is the workaround for the pre-LN issue of redundant LayerNorm.
        # We call here, to avoid fiddling with wav2vec2's model and config.
        if self.layer_norm is not None:
            encoder_output = self.layer_norm(encoder_output)

        encoder_output = self.final_dropout(encoder_output)
        encoder_output_pooled = self.encoder_pooler(
            encoder_output, encoder_padding_mask
        )

        return SonarSpeechEncoderOutput(
            text_embeddings=target_embeddings,
            speech_embeddings=encoder_output_pooled,
            padding_mask=padding_mask,
        )

    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        sonar_output_encoder = self.encoder(seqs, padding_mask)
        return (
            sonar_output_encoder.sentence_embeddings.unsqueeze(1),
            None,
        )

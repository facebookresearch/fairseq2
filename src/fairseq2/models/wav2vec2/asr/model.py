# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout
from torch.nn.functional import ctc_loss, log_softmax

from fairseq2.data import VocabularyInfo
from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn import Linear
from fairseq2.nn.padding import PaddingMask, get_seq_lens, pad_seqs
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device


@final
class Wav2Vec2AsrModel(Model):
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
            init_fn=init_final_projection,
            device=device,
            dtype=dtype,
        )

        self.target_vocab_info = target_vocab_info

    def forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
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

        return Wav2Vec2AsrOutput(logits, padding_mask)


def init_final_projection(proj: Linear) -> None:
    """Initialize ``proj`` as the final projection of a wav2vec 2.0 ASR model."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


@final
@dataclass
class Wav2Vec2AsrOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},T)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, and :math:`T` is the size of the vocabulary."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`logits`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: PaddingMask | None
    ) -> Tensor:
        """Compute the CTC (Connectionist Temporal Classification) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """
        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        lprobs_t = lprobs.transpose(0, 1)

        # (N)
        seq_lens = get_seq_lens(lprobs, self.padding_mask)

        # (N)
        target_seq_lens = get_seq_lens(targets, target_padding_mask)

        # ()
        return ctc_loss(
            lprobs_t,
            targets,
            seq_lens,
            target_seq_lens,
            reduction="sum",
            zero_infinity=True,
        )

    def generate_hypotheses(
        self, pad_idx: int, blank_label: int = 0
    ) -> tuple[Tensor, PaddingMask | None]:
        """Generate hypotheses using greedy search.

        :param pad_idx:
            The index of the PAD symbol in the target vocabulary.
        :param blank_label:
            The blank label in logits.

        :returns:
            - The generated token (i.e. unit) sequences. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the sequence
              length.
            - The padding mask of the generated sequences. *Shape:* Same as the
              generated sequences.
        """
        seq_lens = get_seq_lens(self.logits, self.padding_mask)

        hyp_seq_list = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, seq_len in zip(self.logits, seq_lens):
            # (S)
            hyp_seq = logits[:seq_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != blank_label]

            hyp_seq_list.append(hyp_seq)

        # (N, S), (N, S)
        return pad_seqs(hyp_seq_list, pad_value=pad_idx)

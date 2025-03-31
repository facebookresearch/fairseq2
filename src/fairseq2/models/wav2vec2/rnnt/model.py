# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import final

import numpy as np

import torch
import torch.nn as nn
from fairseq2.data import VocabularyInfo
from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerEmbeddingFrontend
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker

from fairseq2.models.wav2vec2.rnnt.beam_search_gpu import RnntBeamSearchModule
from fairseq2.nn.padding import get_seq_lens, pad_seqs, PaddingMask
from fairseq2.nn.transformer import TransformerEncoder
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Dropout

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchaudio.functional import rnnt_loss


@final
class Wav2Vec2RnntModel(Model):
    """Represents a wav2vec 2.0 encoder as an encoder of an RNNT model."""

    model_dim: int
    encoder: nn.Module
    encoder_frontend: Wav2Vec2Frontend
    text_frontend: nn.Module
    predictor: nn.Module
    final_proj: nn.Module
    masker: Wav2Vec2Masker | None
    final_dropout: Dropout | None
    target_vocab_info: VocabularyInfo

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        text_frontend: TransformerEmbeddingFrontend,
        predictor: torch.nn.Module,
        joiner: torch.nn.Module,
        target_vocab_info: VocabularyInfo,
        *,
        masker: Wav2Vec2Masker | None = None,
        final_dropout_p: float = 0.0,
        beam_search_config: object = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :text_frontend:
            The embedding module for text tokens.
        :param predictor:
            The RNN-T predictor.
        :param joiner:
            The RNN-T joiner.
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
        self.text_frontend = text_frontend
        self.predictor = predictor

        self.register_module("masker", masker)

        if final_dropout_p > 0.0:
            self.final_dropout = Dropout(final_dropout_p)
        else:
            self.register_module("final_dropout", None)

        # Build the joiner
        self.blank = (
            0  # Index 0 is bos_idx which is not used, so repurpose it for blank
        )
        self.final_proj = joiner

        self.target_vocab_info = target_vocab_info
        self.beam_search_config = beam_search_config

    def joiner(self, encoder_out, predictor_out):
        """
        :param encoder_out:
            The output of the encoder. *Shape:* :math:`(N,S_{out},D)`.
        :param predictor_out:
            The output of the predictor. *Shape:* :math:`(N,S_{tgt},D)`.
        """
        joiner_input = encoder_out.unsqueeze(2) + predictor_out.unsqueeze(1)
        logits = self.final_proj(joiner_input)
        return logits  # [B, S, U, D]

    def forward(self, batch: SequenceBatch) -> Wav2Vec2RnntOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        # Run the encoder
        seqs, padding_mask, _ = self.encoder_frontend.extract_features(
            batch.source_seqs, batch.source_padding_mask
        )
        seqs, padding_mask, _ = self.encoder_frontend.process_features(
            seqs, padding_mask, self.masker if self.training else None
        )
        seqs, padding_mask = self.encoder(seqs, padding_mask)

        if self.final_dropout is not None:
            seqs = self.final_dropout(seqs)

        # Prepend blank
        targets_with_blank = torch.cat(
            [
                torch.full_like(batch.target_seqs[:, :1], fill_value=self.blank),
                batch.target_seqs,
            ],
            dim=-1,
        )
        if batch.target_padding_mask is not None:
            target_lengths = batch.target_padding_mask.seq_lens
        else:
            target_lengths = torch.full_like(
                batch.target_seqs[:, 0], fill_value=batch.target_seqs.size(1)
            )
        lengths_with_blank = target_lengths + 1

        # Run the predictor
        predictor_out = self.text_frontend(targets_with_blank)
        predictor_out = pack_padded_sequence(
            predictor_out,
            lengths_with_blank.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        predictor_out, _ = self.predictor(predictor_out)
        predictor_out, _ = pad_packed_sequence(predictor_out, batch_first=True)

        # Run the joiner
        logits = self.joiner(seqs, predictor_out)  # [B, S, U, D]

        return Wav2Vec2RnntOutput(
            logits, seqs, padding_mask, self.blank, self, self.beam_search_config
        )


@final
@dataclass
class Wav2Vec2RnntOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},T, V)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, :math:`T` is the length of the text sequence, :math:`V` is the size
    of the vocabulary."""

    encoder_output: Tensor
    """
    The output of the audio encoder. *Shape:* :math:`(N,S_{out},D)`.
    """

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`logits`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    blank: int
    """The index of the blank symbol in the vocabulary."""

    model: nn.Module
    """A reference to the model."""

    beam_search_config: object
    """Hyperparameters for RNN-T beam search."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: PaddingMask | None
    ) -> Tensor:
        """Compute the RNN-T loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """

        # torchaudio's rnnt_loss has a bug, it throws a CUDA exception when the
        # total number of elements in the logits tensor > 2 ** 31. We split the
        # batch to overcome this.
        # https://github.com/pytorch/audio/issues/3736

        # We need lengths for the rnnt_loss
        if self.padding_mask is not None:
            logit_lengths = self.padding_mask.seq_lens
        else:
            logit_lengths = torch.full_like(
                targets[:, 0], fill_value=self.logits.size(1)
            )
        if target_padding_mask is not None:
            target_lengths = target_padding_mask.seq_lens
        else:
            target_lengths = torch.full_like(targets[:, 0], fill_value=targets.size(1))

        # rnnt_loss expects the logits to be in float16/32
        if self.logits.dtype not in [torch.float32, torch.float16]:
            self.logits = self.logits.float()

        if self.logits.numel() < 2**31:
            loss = rnnt_loss(
                logits=self.logits.float(),
                targets=targets.int(),
                logit_lengths=logit_lengths.int(),
                target_lengths=target_lengths.int(),
                blank=self.blank,
                reduction="sum",
            )
        else:
            # Determine splits
            # numel(b) = b / B * numel(B) < 2 ** 31 --> b < B * 2 ** 31 / numel(B)
            split_size = int(self.logits.size(0) * (2**31) / self.logits.numel())
            if split_size == 0:
                # In that case we want to skip this batch alltogether.
                # The below form makes sure gradients are not None.
                return (self.logits * 0.0).sum()
            num_splits = math.ceil(self.logits.size(0) / split_size)

            losses = []
            for i in range(num_splits):
                # Set start and end indices
                start = i * split_size
                end = start + split_size
                if i == num_splits - 1:
                    end = self.logits.size(0)

                # Trim all tensors and compute loss
                max_source_length = logit_lengths[start:end].max()
                max_target_length = target_lengths[start:end].max()
                new_logits = self.logits[
                    start:end, :max_source_length, : max_target_length + 1
                ].contiguous()
                assert new_logits.numel() < 2**31

                loss = rnnt_loss(
                    logits=new_logits,
                    targets=targets[start:end, :max_target_length].contiguous().int(),
                    logit_lengths=logit_lengths[start:end].int(),
                    target_lengths=target_lengths[start:end].int(),
                    blank=self.blank,
                    reduction="sum",
                )
                losses.append(loss)
            loss = sum(losses)

        return loss

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

        # Make sure no incositencies in config
        assert blank_label == self.blank

        # Lazy creation of the beam search module
        if not hasattr(self, "beam_search_decoder"):
            self.beam_search_decoder = RnntBeamSearchModule(
                text_frontend=self.model.text_frontend,
                predictor=self.model.predictor,
                joiner=self.model.joiner,
                blank=self.blank,
                step_max_symbols=self.beam_search_config.step_max_symbols,
                vocab_size=self.logits.size(-1),
                length_norm=self.beam_search_config.length_norm,
                merge_beam=self.beam_search_config.merge_beam,
                always_merge_blank=self.beam_search_config.always_merge_blank,
            )

        # Run beam search
        if self.padding_mask is not None:
            encoder_lengths = self.padding_mask.seq_lens
        else:
            encoder_lengths = torch.full_like(
                self.encoder_output[:, 0, 0], fill_value=self.encoder_output.size(1)
            )
        tokens, _, _ = self.beam_search_decoder(
            encoder_out=self.encoder_output,
            encoder_lengths=encoder_lengths,
            nbest=self.beam_search_config.nbest,
        )

        # Get best1 tokens only
        B = tokens.size(0)
        best1_hypos = tokens[:, 0, :]
        valid_tokens_mask = torch.logical_and(
            best1_hypos >= 0, best1_hypos < self.logits.size(-1)
        )
        valid_tokens_count = valid_tokens_mask.sum(dim=1)
        max_len = valid_tokens_count.max()
        final_tokens = torch.full(
            [B, max_len], fill_value=pad_idx, dtype=torch.int64, device=tokens.device
        )
        for i in range(B):
            final_tokens[i, : valid_tokens_count[i]] = best1_hypos[i][
                valid_tokens_mask[i]
            ]
        padding_mask = PaddingMask(valid_tokens_count, max_len)

        return final_tokens, padding_mask

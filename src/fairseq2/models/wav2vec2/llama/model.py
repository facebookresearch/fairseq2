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
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    TransformerEmbeddingFrontend,
)
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.typing import DataType, Device

from torch import Tensor
from torch.nn import Dropout


@final
class Wav2Vec2LlamaModel(Model):
    """Represents a wav2vec 2.0 encoder feeding to a Llama decoder for ASR."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    encoder_proj: nn.Module
    text_frontend: TransformerEmbeddingFrontend
    llama_decoder: TransformerDecoder
    final_proj: nn.Module
    masker: Wav2Vec2Masker | None
    final_dropout: Dropout | None
    target_vocab_info: VocabularyInfo

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        encoder_proj: nn.Module,
        text_frontend: TransformerEmbeddingFrontend,
        llama_decoder: TransformerDecoder,
        final_proj: nn.Module,
        target_vocab_info: VocabularyInfo,
        *,
        masker: Wav2Vec2Masker | None = None,
        final_dropout_p: float = 0.0,
        max_generation_length: int = 8192,
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
        self.encoder_proj = encoder_proj
        self.text_frontend = text_frontend
        self.llama_decoder = llama_decoder
        self.final_proj = final_proj

        self.register_module("masker", masker)

        if final_dropout_p > 0.0:
            self.final_dropout = Dropout(final_dropout_p)
        else:
            self.register_module("final_dropout", None)

        self.target_vocab_info = target_vocab_info
        self.max_generation_length = max_generation_length

        self._device = device
        self._dtype = dtype

    def forward(self, batch: SequenceBatch) -> Wav2Vec2LlamaOutput:
        """
        :param batch:
            The batch of sequences to process.
        """
        # Run the encoder
        enc_out, enc_padding_mask, _ = self.encoder_frontend.extract_features(
            batch.source_seqs, batch.source_padding_mask
        )
        enc_out, enc_padding_mask, _ = self.encoder_frontend.process_features(
            enc_out, enc_padding_mask, self.masker if self.training else None
        )
        enc_out, enc_padding_mask = self.encoder(enc_out, enc_padding_mask)

        if self.final_dropout is not None:
            enc_out = self.final_dropout(enc_out)

        # Project encoder outputs to decoder input dimension
        enc_out = self.encoder_proj(enc_out)

        # Prepand BOS and append EOS to text tokens
        targets = torch.cat(
            [
                torch.full_like(
                    batch.target_seqs[:, :1], fill_value=self.target_vocab_info.bos_idx
                ),
                batch.target_seqs,
                torch.full_like(
                    batch.target_seqs[:, :1], fill_value=self.target_vocab_info.pad_idx
                ),
            ],
            dim=-1,
        )
        targets[
            torch.arange(targets.size(0)), batch.target_padding_mask.seq_lens + 1
        ] = self.target_vocab_info.eos_idx
        target_padding_mask = PaddingMask(
            batch.target_padding_mask.seq_lens + 2,
            batch.target_padding_mask.seq_lens.max() + 2,
        )

        # Embed text tokens
        text_embedded, _ = self.text_frontend(targets, target_padding_mask)

        # Prepare decoder input: audio encodings, BOS, text embeddings, EOS
        # Push all paddings to the end
        final_lengths = enc_padding_mask.seq_lens + target_padding_mask.seq_lens
        decoder_inputs = torch.zeros(
            [enc_out.size(0), final_lengths.max(), enc_out.size(2)],
            device=enc_out.device,
            dtype=enc_out.dtype,
        )
        decoder_inputs_padding_mask = PaddingMask(final_lengths, final_lengths.max())
        decoder_inputs[:, : enc_out.size(1)] = enc_out
        for i in range(enc_out.size(0)):
            decoder_inputs[i, enc_padding_mask.seq_lens[i] : final_lengths[i]] = (
                text_embedded[i, : target_padding_mask.seq_lens[i]]
            )

        # Run the decoder
        dec_out, _ = self.llama_decoder(decoder_inputs, decoder_inputs_padding_mask)
        logits = self.final_proj(dec_out)

        return Wav2Vec2LlamaOutput(
            logits,
            enc_out,
            enc_padding_mask,
            self,
            pad_idx=self.target_vocab_info.pad_idx,
            eos_idx=self.target_vocab_info.eos_idx,
        )


@final
@dataclass
class Wav2Vec2LlamaOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out}, V)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the decoder sequence
    length, :math:`V` is the size
    of the vocabulary."""

    enc_out: Tensor
    """
    The output of the audio encoder. *Shape:* :math:`(N,S_{out},D)`.
    """

    enc_padding_mask: PaddingMask | None
    """The padding mask of the audio encoder outputs. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` a sequence
    length."""

    model: nn.Module
    """A reference to the model."""

    pad_idx: int
    """The index of the padding symbol in the target vocabulary."""

    eos_idx: int
    """The index of the end-of-sequence symbol in the target vocabulary."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: PaddingMask | None
    ) -> Tensor:
        """Compute the loss for the speech llama model.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A tensor representing the loss per sample in the batch.
        """

        # Add EOS to the targets
        targets = torch.cat(
            [
                targets,
                torch.full_like(targets[:, :1], fill_value=self.pad_idx),
            ],
            dim=-1,
        )
        targets[torch.arange(targets.size(0)), target_padding_mask.seq_lens] = (
            self.eos_idx
        )

        # Choose the indieces BOS : BOS + max_target_length
        logits_no_enc = torch.zeros_like(
            self.logits[:, : targets.size(1), :],
        )
        for i in range(self.logits.size(0)):
            enc_len_i = self.enc_padding_mask.seq_lens[i]
            tgt_len_i = target_padding_mask.seq_lens[i] + 1  # +1 for EOS
            logits_no_enc[i, :tgt_len_i] = self.logits[
                i, enc_len_i : enc_len_i + tgt_len_i
            ]

        # Run CE loss
        loss = torch.nn.functional.cross_entropy(
            input=logits_no_enc.transpose(1, 2),
            target=targets,
            ignore_index=self.pad_idx,
            reduction="sum",
        )

        # Average per token, but multiple by the number of samples in the batch,
        # Resulting in the required summed loss across the batch, but still considering
        # every token equally in the batch (no advantage to shorter sequences)
        loss = loss / (target_padding_mask.seq_lens + 1).sum() * targets.size(0)
        return loss

    def generate_hypotheses(
        self, pad_idx: int, blank_label: int = 0
    ) -> tuple[Tensor, PaddingMask | None]:

        # Prepare a decoder input matrix, prefill with encoder outputs
        B = self.enc_out.size(0)
        decoder_inputs = torch.zeros(
            [
                B,
                self.model.max_generation_length,
                self.model.llama_decoder.model_dim,
            ],
            device=self.enc_out.device,
            dtype=self.enc_out.dtype,
        )
        decoder_inputs[:, : self.enc_out.size(1)] = self.enc_out

        # Prepare a token output matrix,
        out_tokens = torch.full_like(
            decoder_inputs[:, :, 0],
            fill_value=pad_idx,
            dtype=torch.int,
        )

        # Embed BOS and add to decoder input matrix
        bos_embedded, _ = self.model.text_frontend(
            torch.full(
                [B, 1],
                fill_value=self.model.target_vocab_info.bos_idx,
                dtype=torch.int,
                device=self.enc_out.device,
            ),
            padding_mask=None,
        )
        enc_lengths = self.enc_padding_mask.seq_lens
        decoder_inputs[torch.arange(B), enc_lengths] = bos_embedded.to(
            decoder_inputs.dtype
        ).squeeze(1)

        # Prefill with shortest encoder outputs, keep state
        state_bag = IncrementalStateBag(max_num_steps=self.model.max_generation_length)
        min_enc_len = enc_lengths.min()
        _, _ = self.model.llama_decoder(
            seqs=decoder_inputs[:, :min_enc_len],
            padding_mask=None,
            state_bag=state_bag,
        )

        def combine_masks(mask1, mask2):
            combined_mask = torch.zeros_like(mask1)
            combined_mask[mask1] = mask2
            return combined_mask

        # Iterative decoding
        # For each sample, choose either encooder output, or text embedding
        # If EOS is emitted, the sample is non-active
        # Stop when there are no active samples
        active_mask = torch.ones_like(enc_lengths, dtype=torch.bool)
        done = False
        t = min_enc_len
        while not done:
            # Run the decoder on mixed encoder outputs and text embeddings
            dec_out, _ = self.model.llama_decoder(
                seqs=decoder_inputs[active_mask, t : t + 1],
                padding_mask=None,
                state_bag=state_bag,
            )

            # For samples that had token inputs, embed them and place back in decoder input
            token_mask = enc_lengths[active_mask] <= t
            logits = self.model.final_proj(dec_out[token_mask])
            new_tokens = torch.argmax(logits.squeeze(1), dim=-1)
            out_tokens[combine_masks(active_mask, token_mask), t] = new_tokens.int()

            # Run new tokens through frontend, set in decoder input
            new_tokens_embedded, _ = self.model.text_frontend(
                new_tokens.unsqueeze(1),
                padding_mask=None,
            )
            decoder_inputs[combine_masks(active_mask, token_mask), t + 1] = (
                new_tokens_embedded.to(decoder_inputs.dtype).squeeze(1)
            )

            # If emitted BOS, change active mask and shrink the state
            eos_mask = new_tokens == self.model.target_vocab_info.eos_idx
            newly_inactive_mask = combine_masks(token_mask, eos_mask)
            active_mask[combine_masks(active_mask, newly_inactive_mask)] = False
            state_bag.reorder(torch.where(torch.logical_not(newly_inactive_mask))[0])

            # Decide if we are done
            done = torch.logical_or(
                torch.all(torch.logical_not(active_mask)),
                t == self.model.max_generation_length - 2,
            )
            t += 1

        # Get final tokens
        valid_tokens_mask = torch.logical_and(
            torch.logical_and(
                out_tokens != pad_idx,
                out_tokens != self.model.target_vocab_info.bos_idx,
            ),
            out_tokens != self.model.target_vocab_info.eos_idx,
        )
        valid_tokens_count = valid_tokens_mask.sum(dim=1)
        final_tokens = torch.full(
            [B, valid_tokens_count.max()],
            fill_value=pad_idx,
            dtype=torch.int64,
            device=out_tokens.device,
        )
        for i in range(B):
            final_tokens[i, : valid_tokens_count[i]] = out_tokens[i][
                valid_tokens_mask[i]
            ]
        padding_mask = PaddingMask(valid_tokens_count, valid_tokens_count.max())

        return final_tokens, padding_mask

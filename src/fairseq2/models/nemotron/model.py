# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal NemotronH model wrapping the text-only TransformerLM with audio.

TransformerLM is ``@final`` so we cannot subclass it. Instead, this module
holds the LM as a sub-module and calls its components directly:

    decoder_frontend → [replace audio tokens] → decoder → final_proj

Audio processing:
    mel_features → sound_encoder → sound_projection → audio_embeds
    input_embeds[sound_mask] = audio_embeds  (replace placeholder tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final, overload

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.models.clm import CausalLM
from fairseq2.models.nemotron.audio.conformer import ParakeetAudioTower
from fairseq2.models.nemotron.audio.projection import SoundProjection
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.nn.functional import cross_entropy


@final
class NemotronHMultimodalModel(CausalLM):
    """NemotronH with optional audio (Parakeet) encoder.

    When ``sound_encoder`` and ``sound_projection`` are present, audio features
    replace ``<so_embedding>`` placeholder tokens in the input sequence before
    decoding. When no audio is provided, behaves identically to a text-only
    TransformerLM.

    Note: TransformerLM is ``@final``, so this is a wrapper that holds it as
    ``self.language_model`` and calls its sub-components directly.
    """

    def __init__(
        self,
        language_model: TransformerLM,
        sound_encoder: ParakeetAudioTower | None = None,
        sound_projection: SoundProjection | None = None,
        sound_context_token_id: int = 27,
    ) -> None:
        super().__init__(language_model.max_seq_len)

        self.language_model = language_model
        self.sound_encoder = sound_encoder
        self.sound_projection = sound_projection
        self.sound_context_token_id = sound_context_token_id

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
        mel_features: Tensor | None = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        mel_features: Tensor | None = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[False],
        mel_features: Tensor | None = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
        mel_features: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @override
    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
        mel_features: Tensor | None = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
        mel_features: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass with optional audio.

        :param seqs:
            Token IDs. *Shape:* ``[B, S]``.
        :param seqs_layout:
            Batch layout for the sequences.
        :param targets:
            Target token IDs for loss computation. *Shape:* ``[B, S]``.
        :param state_bag:
            Incremental state bag for autoregressive generation.
        :param mel_features:
            Mel spectrogram features. *Shape:* ``[B, T, mel_bins]``.
            If ``None``, no audio processing is done.

        :returns:
            Logits ``[B, S, V]`` when targets is None, or loss scalar.
        """
        lm = self.language_model

        # Step 1: Embed text tokens via the LM's frontend
        embedded_seqs, seqs_layout = lm.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        # Step 2: If audio is provided, encode and replace placeholder tokens
        if mel_features is not None and self.sound_encoder is not None and self.sound_projection is not None:
            embedded_seqs = self._inject_audio(
                seqs, embedded_seqs, mel_features
            )

        # Step 3: Run decoder
        decoder_output = lm.decoder(embedded_seqs, seqs_layout, state_bag=state_bag)

        del embedded_seqs

        # Step 4: Compute logits or loss
        if targets is None:
            return lm.final_proj(decoder_output)

        if not return_logits:
            logits = lm.final_proj(decoder_output)
            del decoder_output
            return cross_entropy(
                logits,
                targets,
                lm.pad_idx,
                label_smoothing=label_smoothing,
                target_mask=target_mask,
                reduction=reduction,
            )

        logits = lm.final_proj(decoder_output)
        del decoder_output
        loss = cross_entropy(
            logits,
            targets,
            lm.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )
        return loss, logits

    def _inject_audio(
        self,
        input_ids: Tensor,
        input_embeds: Tensor,
        mel_features: Tensor,
    ) -> Tensor:
        """Encode audio and replace placeholder tokens with audio embeddings.

        :param input_ids:
            Original token IDs. *Shape:* ``[B, S]``.
        :param input_embeds:
            Text embeddings from decoder frontend. *Shape:* ``[B, S, D]``.
        :param mel_features:
            Mel spectrogram features. *Shape:* ``[B, T, mel_bins]``.

        :returns:
            Modified embeddings with audio tokens replaced.
        """
        assert self.sound_encoder is not None
        assert self.sound_projection is not None

        # Encode audio: [B, T, mel_bins] -> [B, T/8, encoder_dim]
        audio_features = self.sound_encoder(mel_features)

        # Project to LM space: [B, T/8, encoder_dim] -> [B, T/8, model_dim]
        audio_embeds = self.sound_projection(audio_features)

        # Find placeholder tokens
        sound_mask = input_ids == self.sound_context_token_id

        # Flatten audio embeddings across batch for replacement
        flat_audio = audio_embeds.reshape(-1, audio_embeds.shape[-1])

        # Replace: input_embeds[sound_mask] = flat_audio
        # Ensure dtype match
        input_embeds = input_embeds.clone()
        num_placeholders = sound_mask.sum().item()
        if num_placeholders > 0:
            input_embeds[sound_mask] = flat_audio[:num_placeholders].to(
                input_embeds.dtype
            )

        return input_embeds

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        has_audio = self.sound_encoder is not None
        return (
            f"max_seq_len={self.max_seq_len}, "
            f"has_audio={has_audio}, "
            f"sound_context_token_id={self.sound_context_token_id}"
        )

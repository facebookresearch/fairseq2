# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 decoder frontend with optional Per-Layer Embeddings (PLE) and
audio injection.

When PLE is disabled (``ple_hidden_dim == 0``), this is a simple embedding
lookup with ``sqrt(model_dim)`` scaling.  When PLE is enabled, it behaves
identically to :class:`~fairseq2.models.gemma3n.frontend.Gemma3nFrontend`
(discrete + continuous per-layer embeddings).

Supports audio injection: when ``audio_embeds`` is provided to ``forward``,
positions matching ``audio_token_id`` in the input are replaced with
pre-encoded audio embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import (
    BatchLayout,
    Embedding,
    IncrementalStateBag,
    LayerNorm,
    StandardEmbedding,
)
from fairseq2.nn.projection import Linear

__all__ = ["Gemma4Frontend"]


class Gemma4Frontend(Module):
    """Gemma 4 decoder frontend with optional PLE and audio injection."""

    embed: Embedding
    scale: float
    audio_token_id: int | None

    # PLE modules (None when PLE is disabled).
    embed_tokens_per_layer: StandardEmbedding | None
    per_layer_model_projection: Linear | None
    per_layer_projection_norm: LayerNorm | None
    num_layers: int
    ple_hidden_dim: int

    def __init__(
        self,
        model_dim: int,
        embed: Embedding,
        *,
        num_layers: int,
        ple_hidden_dim: int = 0,
        vocab_size_per_layer_input: int = 0,
        ple_norm: LayerNorm | None = None,
        audio_token_id: int | None = None,
        pad_idx: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality.
        :param embed: Token embedding table.
        :param num_layers: Number of decoder layers.
        :param ple_hidden_dim: Hidden dim for PLE.  0 disables PLE.
        :param vocab_size_per_layer_input: Vocabulary size for PLE lookup.
        :param ple_norm: RMSNorm for PLE projection (required when PLE enabled).
        :param audio_token_id: Token ID used as placeholder for audio
            embeddings.  ``None`` disables audio injection.
        :param pad_idx: Padding token index.  Used to replace multimodal
            placeholder tokens in discrete PLE, matching HuggingFace.
        :param device: Device.
        :param dtype: Data type.
        """
        super().__init__()

        self.embed = embed
        self.scale = model_dim ** 0.5
        self.num_layers = num_layers
        self.ple_hidden_dim = ple_hidden_dim
        self.audio_token_id = audio_token_id
        self.pad_idx = pad_idx

        if ple_hidden_dim > 0 and vocab_size_per_layer_input > 0:
            # PLE enabled.
            self.embed_tokens_per_layer = StandardEmbedding(
                num_embeddings=vocab_size_per_layer_input,
                embed_dim=num_layers * ple_hidden_dim,
                pad_idx=None,
                device=device,
                dtype=dtype,
            )

            self.per_layer_model_projection = Linear(
                model_dim,
                num_layers * ple_hidden_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )

            if ple_norm is None:
                raise ValueError(
                    "`ple_norm` must be provided when PLE is enabled."
                )
            self.per_layer_projection_norm = ple_norm

            # Scaling buffers (non-persistent).
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(model_dim ** -0.5, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_embed_scale",
                torch.tensor(ple_hidden_dim ** 0.5, device=device, dtype=dtype),
                persistent=False,
            )
        else:
            # PLE disabled.
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        audio_embeds: Tensor | None = None,
        vision_features: Tensor | None = None,
    ) -> tuple[Tensor, BatchLayout, Tensor | None]:
        """
        :param seqs: Token IDs. *Shape:* ``(B, S)``.
        :param seqs_layout: Layout information.
        :param state_bag: Incremental decoding state.
        :param audio_embeds: Pre-encoded audio embeddings from
            ``audio_tower + audio_embedder``.  *Shape:* ``(B, T_a, M)``.
        :param vision_features: Unused (Gemma 4 has no vision tower).
        :returns:
            - Embeddings ``(B, S, M)``
            - Layout
            - Per-layer embeddings ``(B, S, L, ple_dim)`` or ``None``
        """
        token_ids = seqs

        seqs = self.embed(seqs)
        seqs = seqs * self.scale

        # Inject pre-encoded audio embeddings at audio_token_id positions.
        if audio_embeds is not None and self.audio_token_id is not None:
            seqs = self._inject_audio_embeds(token_ids, seqs, audio_embeds)

        if self.embed_tokens_per_layer is not None:
            # For discrete PLE, replace multimodal placeholder tokens with
            # pad_idx to match HF's behavior (which feeds pad_token_id into
            # embed_tokens_per_layer at multimodal positions).
            ple_token_ids = token_ids
            if (
                audio_embeds is not None
                and self.audio_token_id is not None
                and self.pad_idx is not None
            ):
                ple_token_ids = token_ids.clone()
                ple_token_ids[token_ids == self.audio_token_id] = self.pad_idx
            per_layer_inputs = self._compute_ple(ple_token_ids, seqs)
        else:
            per_layer_inputs = None

        return seqs, seqs_layout, per_layer_inputs

    def _compute_ple(
        self, token_ids: Tensor, seqs: Tensor
    ) -> Tensor:
        """Compute per-layer embeddings (discrete + continuous).

        :param token_ids: Token IDs ``(B, S)``.
        :param seqs: Scaled embeddings ``(B, S, M)``.
        :returns: PLE ``(B, S, num_layers, ple_hidden_dim)``.
        """
        assert self.embed_tokens_per_layer is not None
        assert self.per_layer_model_projection is not None
        assert self.per_layer_projection_norm is not None

        # Discrete PLE.
        ple_token_ids = torch.clamp(
            token_ids, max=self.embed_tokens_per_layer.num_embeddings - 1
        )
        discrete = self.embed_tokens_per_layer(ple_token_ids)  # (B, S, L*P)
        discrete = discrete * self.per_layer_embed_scale  # type: ignore[operator]
        discrete = discrete.reshape(
            *token_ids.shape, self.num_layers, self.ple_hidden_dim
        )

        # Continuous PLE.
        continuous = self.per_layer_model_projection(seqs)  # (B, S, L*P)
        continuous = continuous * self.per_layer_projection_scale  # type: ignore[operator]
        continuous = continuous.reshape(
            *seqs.shape[:-1], self.num_layers, self.ple_hidden_dim
        )
        continuous = self.per_layer_projection_norm(continuous)

        # Combine.
        scale = self.per_layer_input_scale  # type: ignore[assignment]
        return (continuous + discrete) * scale

    def _inject_audio_embeds(
        self,
        token_ids: Tensor,
        text_embeds: Tensor,
        audio_embeds: Tensor,
    ) -> Tensor:
        """Replace audio placeholder token embeddings with encoded audio.

        :param token_ids: Token IDs. *Shape:* ``(B, S)``.
        :param text_embeds: Text embeddings. *Shape:* ``(B, S, M)``.
        :param audio_embeds: Audio features. *Shape:* ``(B, T_a, M)``.
        :returns: Merged embeddings. *Shape:* ``(B, S, M)``.
        """
        assert self.audio_token_id is not None

        # Boolean mask: True where the token is an audio placeholder.
        mask = token_ids == self.audio_token_id  # (B, S)

        result = text_embeds.clone()
        for i in range(token_ids.size(0)):
            n_slots = mask[i].sum().item()
            if n_slots == 0:
                continue

            n_frames = audio_embeds.size(1)

            if n_frames >= n_slots:
                # Enough frames — take the first n_slots.
                result[i, mask[i]] = audio_embeds[i, :n_slots]
            else:
                # Fewer frames than slots — pad with zeros.
                padded = torch.cat(
                    [
                        audio_embeds[i, :n_frames],
                        audio_embeds.new_zeros(n_slots - n_frames, audio_embeds.size(2)),
                    ],
                    dim=0,
                )
                result[i, mask[i]] = padded

        return result

    if TYPE_CHECKING:
        __call__ = forward

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast, final

import torch
from torch import Tensor
from torch.nn import Dropout, Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import (
    BatchLayout,
    Embedding,
    IncrementalStateBag,
    LayerNorm,
    PositionEncoder,
    StandardEmbedding,
)
from fairseq2.nn.projection import Linear


class Gemma3nFrontendBase(Module, ABC):
    """Base class for Gemma3n frontends with multimodal support."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        audio_embeds: Tensor | None = None,
        vision_features: Tensor | None = None,
    ) -> tuple[Tensor, BatchLayout, Tensor]:
        """
        :param seqs: Token IDs. *Shape:* :math:`(N,S)`.
        :param seqs_layout: Layout information for ``seqs``.
        :param state_bag: Incremental decoding state (KV cache).
        :param audio_embeds: Pre-encoded audio. *Shape:* :math:`(N,T,M)`.
        :param vision_features: Image pixel values. *Shape:* :math:`(N,C,H,W)`.
        :returns:
            - Embeddings *Shape:* :math:`(N,S,M)`
            - Layout
            - Per-layer embeddings *Shape:* :math:`(N,S,L,H)`
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class Gemma3nFrontend(Gemma3nFrontendBase):
    """Gemma3n frontend with PLE (Per-Layer Embeddings) support."""

    embed: Embedding
    pos_encoder: PositionEncoder | None
    dropout: Dropout | None
    scale: float
    embed_tokens_per_layer: StandardEmbedding
    per_layer_model_projection: Linear
    per_layer_projection_norm: LayerNorm
    num_layers: int
    ple_hidden_dim: int
    audio_token_id: int | None
    num_audio_tokens: int

    def __init__(
        self,
        model_dim: int,
        embed: Embedding,
        pos_encoder: PositionEncoder | None,
        *,
        vocab_size_per_layer: int,
        num_layers: int,
        ple_hidden_dim: int,
        ple_norm: LayerNorm,
        no_scale: bool = False,
        dropout_p: float = 0.0,
        audio_token_id: int | None = None,
        num_audio_tokens: int = 188,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality.
        :param embed: Token embedding table.
        :param pos_encoder: Position encoder.
        :param vocab_size_per_layer: Vocabulary size for PLE embeddings.
        :param num_layers: Number of decoder layers.
        :param ple_hidden_dim: Hidden dimension for PLE embeddings.
        :param ple_norm: Layer normalization for PLE.
        :param no_scale: If True, does not scale embeddings.
        :param dropout_p: Dropout probability on embeddings.
        :param audio_token_id: Token ID for <audio> placeholder tokens.
        :param num_audio_tokens: Fixed number of <audio> tokens per audio input.
        """
        super().__init__()

        self.embed = embed
        self.scale = 1.0 if no_scale else (model_dim**0.5)

        self.pos_encoder: PositionEncoder | None
        self.register_module("pos_encoder", pos_encoder)

        if dropout_p > 0.0:
            dropout = Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: Dropout | None
        self.register_module("dropout", dropout)

        self.num_layers = num_layers
        self.ple_hidden_dim = ple_hidden_dim

        self.audio_token_id = audio_token_id
        self.num_audio_tokens = num_audio_tokens

        # PLE: Discrete embedding lookup (shared across all layers)
        self.embed_tokens_per_layer = StandardEmbedding(
            num_embeddings=vocab_size_per_layer,
            embed_dim=num_layers * ple_hidden_dim,
            pad_idx=None,
            device=device,
            dtype=dtype,
        )

        # PLE: Continuous projection from current embeddings
        self.per_layer_model_projection = Linear(
            model_dim,
            num_layers * ple_hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # PLE: Normalization after projection
        self.per_layer_projection_norm = ple_norm

        # PLE: Scaling factors (buffers, not parameters)
        self.register_buffer(
            "per_layer_projection_scale",
            torch.tensor(model_dim**-0.5, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "per_layer_input_scale",
            torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype)),
            persistent=False,
        )
        self.register_buffer(
            "per_layer_embed_scale",
            torch.tensor(ple_hidden_dim**0.5, device=device, dtype=dtype),
            persistent=False,
        )

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        audio_embeds: Tensor | None = None,
        vision_features: Tensor | None = None,
    ) -> tuple[Tensor, BatchLayout, Tensor]:
        """
        :param seqs: Token IDs. *Shape:* :math:`(N,S)`.
        :param seqs_layout: Layout information.
        :param state_bag: Incremental decoding state.
        :param audio_embeds: Pre-encoded audio. *Shape:* :math:`(N,T,M)`.
        :param vision_features: Image pixels. *Shape:* :math:`(N,C,H,W)`.
        :returns:
            - Embeddings *Shape:* :math:`(N,S,2048)`
            - Layout
            - Per-layer embeddings *Shape:* :math:`(N,S,L,256)`
        """
        token_ids = seqs

        seqs = self.embed(seqs)

        if self.scale != 1.0:
            seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, seqs_layout, state_bag=state_bag)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        # Inject pre-encoded audio embeddings
        if audio_embeds is not None and self.audio_token_id is not None:
            seqs = self._inject_audio_embeds(token_ids, seqs, audio_embeds)

        per_layer_inputs_discrete = self._get_per_layer_inputs(token_ids)
        per_layer_inputs_continuous = self._project_per_layer_inputs(seqs)

        per_layer_inputs = self._combine_per_layer_inputs(
            per_layer_inputs_discrete,
            per_layer_inputs_continuous,
        )

        return seqs, seqs_layout, per_layer_inputs

    def _inject_audio_embeds(
        self,
        token_ids: Tensor,
        text_embeds: Tensor,
        audio_embeds: Tensor,
    ) -> Tensor:
        """Replace <audio> token embeddings with pre-encoded audio features.

        Pads audio_embeds with zeros if fewer frames than token slots.

        :param token_ids: Token IDs. *Shape:* :math:`(N,S)`.
        :param text_embeds: Text embeddings. *Shape:* :math:`(N,S,M)`.
        :param audio_embeds: Audio features. *Shape:* :math:`(N,T,M)`.
        :returns: Merged embeddings. *Shape:* :math:`(N,S,M)`.
        """
        assert self.audio_token_id is not None

        upper = self.audio_token_id + self.num_audio_tokens
        mask = (token_ids >= self.audio_token_id) & (token_ids < upper)
        n_slots = self.num_audio_tokens
        n_frames = audio_embeds.size(1)

        # Safeguard: pad if tower produces fewer frames than token slots.
        # The dataloader should handle this, but protect against mismatches.
        if n_frames < n_slots:
            pad = audio_embeds.new_zeros(
                audio_embeds.size(0),
                n_slots - n_frames,
                audio_embeds.size(2),
            )
            audio_embeds = torch.cat([audio_embeds, pad], dim=1)

        result = text_embeds.clone()
        for i in range(token_ids.size(0)):
            result[i, mask[i]] = audio_embeds[i, :n_slots]

        return result

    def _get_per_layer_inputs(self, token_ids: Tensor) -> Tensor:
        """Lookup PLE embeddings from token_ids.

        :param token_ids: Token IDs [B, S].
        :returns: PLE embeddings [B, S, num_layers, ple_hidden_dim].
        """
        # Clip token IDs to valid vocab range for PLE (text-only vocab)
        # Audio/vision tokens are out of PLE vocab range, so use pad token instead
        # (they'll be replaced by tower embeddings in main embedding space anyway)
        ple_token_ids = torch.clamp(
            token_ids, max=self.embed_tokens_per_layer.num_embeddings - 1
        )

        # Lookup from shared embedding table
        per_layer_embeds = self.embed_tokens_per_layer(ple_token_ids)  # [B, S, L*P]

        # Scale embeddings (like Gemma3nTextScaledWordEmbedding)
        per_layer_embeds = per_layer_embeds * self.per_layer_embed_scale  # type: ignore[operator]

        # Reshape to separate layers
        return per_layer_embeds.reshape(
            *token_ids.shape,
            self.num_layers,
            self.ple_hidden_dim,
        )  # [B, S, L, P]

    def _project_per_layer_inputs(self, seqs: Tensor) -> Tensor:
        """Project embeddings to PLE space.

        :param seqs: Embeddings [B, S, M].
        :returns: PLE projections [B, S, num_layers, ple_hidden_dim].
        """
        # Linear projection
        per_layer_proj = self.per_layer_model_projection(seqs)  # [B, S, L*P]

        # Scale
        per_layer_proj = per_layer_proj * self.per_layer_projection_scale  # type: ignore[operator]

        # Reshape to separate layers
        per_layer_proj = per_layer_proj.reshape(
            *seqs.shape[:-1],
            self.num_layers,
            self.ple_hidden_dim,
        )  # [B, S, L, P]

        # Normalize (matches HF line 1772)
        return self.per_layer_projection_norm(per_layer_proj)

    def _combine_per_layer_inputs(
        self,
        discrete: Tensor,
        continuous: Tensor,
    ) -> Tensor:
        """Combine discrete and continuous PLE embeddings.

        :param discrete: Discrete PLE [B, S, L, P].
        :param continuous: Continuous PLE (already normalized) [B, S, L, P].
        :returns: Combined PLE [B, S, L, P].
        """
        # Add and scale (matches HF line 1781)
        scale = cast(Tensor, self.per_layer_input_scale)
        return (continuous + discrete) * scale

    def reset_non_persistent_buffers(self) -> None:
        """Reset non-persistent buffers to their default values."""
        model_dim = self.embed.embed_dim
        self.per_layer_projection_scale.fill_(model_dim**-0.5)  # type: ignore[operator]
        self.per_layer_input_scale.fill_(2.0**-0.5)  # type: ignore[operator]
        self.per_layer_embed_scale.fill_(self.ple_hidden_dim**0.5)  # type: ignore[operator]

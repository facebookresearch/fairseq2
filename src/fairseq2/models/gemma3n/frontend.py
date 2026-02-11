# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer import TransformerEmbeddingFrontend
from fairseq2.nn import (
    BatchLayout,
    Embedding,
    IncrementalStateBag,
    LayerNorm,
    PositionEncoder,
    StandardEmbedding,
)
from fairseq2.nn.projection import Linear


@final
class Gemma3nFrontend(TransformerEmbeddingFrontend):
    """Gemma3n frontend with PLE (Per-Layer Embeddings) support."""

    embed_tokens_per_layer: StandardEmbedding
    per_layer_model_projection: Linear
    per_layer_projection_norm: LayerNorm
    per_layer_projection_scale: Tensor
    per_layer_input_scale: Tensor
    per_layer_embed_scale: Tensor  # Scale for PLE discrete embeddings
    num_layers: int
    ple_hidden_dim: int

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
        """
        super().__init__(
            model_dim,
            embed,
            pos_encoder,
            no_scale=no_scale,
            dropout_p=dropout_p,
            device=device,
            dtype=dtype,
        )

        self.num_layers = num_layers
        self.ple_hidden_dim = ple_hidden_dim

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
    ) -> tuple[Tensor, BatchLayout]:
        """
        :param seqs: Token IDs [B, S] (before embedding).
        :param seqs_layout: Batch layout information.
        :param state_bag: State bag to store PLE embeddings.
        :returns: Tuple of (embedded sequences [B, S, M], layout).
        """
        # Store token_ids for PLE lookup (before embedding)
        token_ids = seqs

        # Get PLE discrete embeddings from token_ids
        per_layer_inputs_discrete = self._get_per_layer_inputs(token_ids)

        # Normal embedding process (converts token_ids → embeddings)
        seqs, seqs_layout = super().forward(seqs, seqs_layout, state_bag=state_bag)

        # Get PLE continuous projection from embeddings
        per_layer_inputs_continuous = self._project_per_layer_inputs(seqs)

        # Combine discrete + continuous PLE
        per_layer_inputs = self._combine_per_layer_inputs(
            per_layer_inputs_discrete,
            per_layer_inputs_continuous,
        )

        # Store in state_bag for decoder to retrieve
        if state_bag is None:
            state_bag = IncrementalStateBag(max_num_steps=seqs.size(1))

        state_bag.per_layer_inputs = per_layer_inputs

        return seqs, seqs_layout

    def _get_per_layer_inputs(self, token_ids: Tensor) -> Tensor:
        """Lookup PLE embeddings from token_ids.

        :param token_ids: Token IDs [B, S].
        :returns: PLE embeddings [B, S, num_layers, ple_hidden_dim].
        """
        # Lookup from shared embedding table
        per_layer_embeds = self.embed_tokens_per_layer(token_ids)  # [B, S, L*P]

        # Scale embeddings (like Gemma3nTextScaledWordEmbedding)
        per_layer_embeds = per_layer_embeds * self.per_layer_embed_scale

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
        per_layer_proj = per_layer_proj * self.per_layer_projection_scale

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
        return (continuous + discrete) * self.per_layer_input_scale

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import LayerNorm, StandardEmbedding
from fairseq2.nn.projection import Linear


@final
class PerLayerEmbedding(Module):
    """Per-Layer Embedding (PLE) augmentation for Gemma3n.

    PLE provides layer-specific embedding enhancements that are computed from
    a shared embedding table and gated per-layer. This allows the model to
    augment token representations dynamically at each layer.

    Reference: Gemma 3 Technical Report (https://arxiv.org/abs/2503.19786)
    """

    embed_tokens_per_layer: StandardEmbedding
    per_layer_input_gate: Linear
    per_layer_projection: Linear
    post_per_layer_input_norm: LayerNorm

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        model_dim: int,
        *,
        layer_norm: LayerNorm,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param vocab_size: Vocabulary size for PLE embeddings.
        :param hidden_size: Hidden dimension of PLE embeddings.
        :param model_dim: Model dimension to project into.
        :param layer_norm: Post-PLE normalization.
        """
        super().__init__()

        # Shared embedding table across all layers
        self.embed_tokens_per_layer = StandardEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            pad_idx=None,
            device=device,
            dtype=dtype,
        )

        # Gate to control PLE contribution
        self.per_layer_input_gate = Linear(
            model_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        # Project PLE embeddings to model dimension
        self.per_layer_projection = Linear(
            hidden_size, model_dim, bias=False, device=device, dtype=dtype
        )

        self.post_per_layer_input_norm = layer_norm

    @override
    def forward(self, seqs: Tensor, token_ids: Tensor) -> Tensor:
        """
        Apply PLE augmentation to sequences.

        :param seqs: Input sequences. *Shape:* (N, S, M) where N is batch size,
            S is sequence length, M is model dimension.
        :param token_ids: Token IDs for PLE lookup. *Shape:* (N, S).
        :returns: Augmented sequences with PLE contribution. *Shape:* Same as seqs.
        """
        # Look up PLE embeddings
        ple_embeds = self.embed_tokens_per_layer(token_ids)  # (N, S, H)

        # Gate based on current sequence state
        gate = self.per_layer_input_gate(seqs)  # (N, S, H)

        # Apply gating to PLE embeddings
        gated_ple = ple_embeds * gate  # (N, S, H)

        # Project to model dimension
        ple_projection = self.per_layer_projection(gated_ple)  # (N, S, M)

        # Add to input and normalize
        augmented = seqs + ple_projection
        return self.post_per_layer_input_norm(augmented)

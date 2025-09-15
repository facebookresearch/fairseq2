# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.models.llama4.model.vision.embedding import VisionEmbeddings
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import (
    Embedding,
    IncrementalStateBag,
    LayerNorm,
    PositionEncoder,
    Projection,
)
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import LayerNormFactory, create_standard_layer_norm


@dataclass
class MaskedEmbedding:
    embedding: Tensor
    mask: Tensor


class LLaMA4DecoderFrontend(TransformerFrontend):
    """Represents a Llama 4 front-end with different embeddings
    for multiple modalities."""

    embed: Embedding
    vision_embed: VisionEmbeddings | None
    vision_proj: Projection | None
    scale: float
    pos_encoder: PositionEncoder | None
    layer_norm: LayerNorm | None
    dropout: Dropout | None

    def __init__(
        self,
        embed: Embedding,
        vision_embed: VisionEmbeddings | None,
        vision_proj: Projection | None,
        pos_encoder: PositionEncoder | None,
        *,
        no_scale: bool = False,
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        layer_norm_factory: LayerNormFactory | None = None,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param vision_encoder:
            The vision embedder.
        :param vision_proj:
            The vision projection.
        :param pos_encoder:
            The position encoder.
        :param no_scale:
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings before
            dropout.
        :param dropout_p:
            The dropout probability on embeddings.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        model_dim = embed.embed_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.embed = embed
        self.vision_embed = vision_embed
        self.vision_proj = vision_proj

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `embedding_dim` of `embed` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @override
    def forward(
        self,
        seqs: Tensor,  # [batch_size, seq_len] or [batch_size, seq_len, 3]
        padding_mask: PaddingMask | None,
        *,
        state_bag: IncrementalStateBag | None = None,
        image_embedding: MaskedEmbedding | None = None,
    ) -> tuple[Tensor, PaddingMask | None]:
        embeds = self.embed(seqs)

        # early image fusion if relevant embeddings are passed
        if image_embedding is not None and self.vision_proj is not None:
            embeds_image = self.vision_proj(image_embedding.embedding)
            embeds = (
                embeds * ~image_embedding.mask + embeds_image * image_embedding.mask
            )

        if self.scale != 1.0:
            embeds = embeds * self.scale

        if self.pos_encoder is not None:
            embeds = self.pos_encoder(embeds, padding_mask, state_bag=state_bag)

        if self.layer_norm is not None:
            embeds = self.layer_norm(embeds)

        if self.dropout is not None:
            embeds = self.dropout(embeds)

        return embeds, padding_mask

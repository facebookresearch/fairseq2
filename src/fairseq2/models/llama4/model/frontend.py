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
    BatchLayout,
    Embedding,
    IncrementalStateBag,
    Projection,
)


@dataclass
class MaskedEmbedding:
    embedding: Tensor
    mask: Tensor


class Llama4DecoderFrontend(TransformerFrontend):
    """Represents a Llama 4 front-end with different embeddings
    for multiple modalities."""

    embed: Embedding
    vision_embed: VisionEmbeddings | None
    vision_proj: Projection | None
    dropout: Dropout | None

    def __init__(
        self,
        embed: Embedding,
        vision_embed: VisionEmbeddings | None,
        vision_proj: Projection | None,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param vision_embed:
            The vision embedder.
        :param vision_proj:
            The vision projection.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings before
            dropout.
        :param dropout_p:
            The dropout probability on embeddings.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        super().__init__()

        self.embed = embed
        self.vision_embed = vision_embed
        self.vision_proj = vision_proj

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        image_embedding: MaskedEmbedding | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        embeds = self.embed(seqs)

        # early image fusion if relevant embeddings are passed
        if image_embedding is not None and self.vision_proj is not None:
            embeds_image = self.vision_proj(image_embedding.embedding)
            embeds = (
                embeds * ~image_embedding.mask + embeds_image * image_embedding.mask
            )

        if self.dropout is not None:
            embeds = self.dropout(embeds)

        return embeds, seqs_layout

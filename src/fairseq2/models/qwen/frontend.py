# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal frontend for Qwen 3.6 VLM.

Takes text token IDs + optional pixel values, embeds text, encodes vision,
and replaces image placeholder tokens with vision embeddings.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from fairseq2.models.qwen.vision_encoder import QwenVisionEncoder, QwenVisionMerger
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import BatchLayout, Embedding, IncrementalStateBag


class Qwen36Frontend(TransformerFrontend):
    """Multimodal frontend: text embedding + vision encoding + feature merging.

    During forward:
    1. Embed all text tokens via the standard embedding table.
    2. If ``pixel_values`` are provided, run vision encoder + merger.
    3. Replace positions of ``image_token_id`` with vision embeddings.
    """

    model_dim: Final[int]
    image_token_id: Final[int]

    def __init__(
        self,
        model_dim: int,
        embed: Embedding,
        vision_encoder: QwenVisionEncoder,
        vision_merger: QwenVisionMerger,
        *,
        image_token_id: int = 248056,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.image_token_id = image_token_id

        self.embed = embed
        self.vision_encoder = vision_encoder
        self.vision_merger = vision_merger
        self.dropout_p = dropout_p

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        """
        Args:
            seqs: Token IDs, shape (B, S).
            seqs_layout: Batch layout.
            state_bag: For incremental decoding.
            pixel_values: Flattened image patches, (total_patches, C*T*P*P).
            image_grid_thw: Grid dimensions per image, (num_images, 3).

        Returns:
            (embeddings, seqs_layout) — embeddings shape (B, S, model_dim).
        """
        # Text embedding
        embeds = self.embed(seqs)

        # Vision encoding
        if pixel_values is not None and image_grid_thw is not None:
            vision_hidden = self.vision_encoder(pixel_values, image_grid_thw)
            vision_embeds = self.vision_merger(vision_hidden, image_grid_thw)

            # Replace image_token_id positions with vision embeddings
            embeds = self._merge_vision_embeddings(
                seqs, embeds, vision_embeds
            )

        return embeds, seqs_layout

    def _merge_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeds: Tensor,
        vision_embeds: Tensor,
    ) -> Tensor:
        """Replace image placeholder tokens with vision embeddings.

        Args:
            input_ids: (B, S) token IDs.
            text_embeds: (B, S, D) text embeddings.
            vision_embeds: (total_vision_tokens, D) merged vision features.

        Returns:
            (B, S, D) with image token positions replaced.
        """
        # Find image token positions
        image_mask = input_ids == self.image_token_id  # (B, S)

        # Flatten for scatter
        B, S, D = text_embeds.shape
        flat_embeds = text_embeds.reshape(-1, D)  # (B*S, D)
        flat_mask = image_mask.reshape(-1)  # (B*S,)

        # Get indices where image tokens are
        image_indices = flat_mask.nonzero(as_tuple=True)[0]

        if image_indices.numel() != vision_embeds.shape[0]:
            raise ValueError(
                f"Number of image token positions ({image_indices.numel()}) "
                f"does not match vision embeddings ({vision_embeds.shape[0]})."
            )

        # Replace with vision embeddings
        flat_embeds = flat_embeds.clone()
        flat_embeds[image_indices] = vision_embeds.to(flat_embeds.dtype)

        return flat_embeds.reshape(B, S, D)

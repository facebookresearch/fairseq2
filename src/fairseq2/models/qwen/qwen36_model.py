# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen 3.6 multimodal VLM model.

Wraps the Qwen 3.5 text backbone with a vision encoder and multimodal frontend.
The text decoder is identical to Qwen 3.5; the only additions are:
  - Vision encoder (ViT) + merger
  - M-RoPE (multimodal rotary position encoding)
  - Qwen36Frontend (replaces TransformerEmbeddingFrontend)
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.models.qwen.mrope import MultimodalRotaryEncoder
from fairseq2.models.transformer_lm import TransformerLMDecoder
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection


class Qwen36Model(Module):
    """Qwen 3.6 Vision-Language Model.

    Architecture:
      - decoder_frontend: Qwen36Frontend (text embed + vision encode + merge)
      - decoder: TransformerLMDecoder (Qwen 3.5 hybrid text backbone)
      - final_proj: Projection to vocab logits
    """

    model_dim: Final[int]
    max_seq_len: Final[int]

    def __init__(
        self,
        model_dim: int,
        decoder_frontend: Module,
        decoder: TransformerLMDecoder,
        final_proj: Projection,
        pos_encoder: MultimodalRotaryEncoder | None,
        *,
        pad_idx: int | None = None,
        max_seq_len: int = 262_144,
        image_token_id: int = 248056,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.image_token_id = image_token_id
        self.mrope_section = mrope_section or [11, 11, 10]
        self.pad_idx = pad_idx

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pos_encoder = pos_encoder

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        Args:
            seqs: Token IDs, (B, S).
            seqs_layout: Batch layout for the token sequence.
            pixel_values: Flattened image patches, (total_patches, C*T*P*P).
            image_grid_thw: Grid dimensions per image, (num_images, 3).
            state_bag: Incremental state for decoding.

        Returns:
            Logits, (B, S, vocab_size).
        """
        # Compute M-RoPE position IDs and set on the position encoder
        if self.pos_encoder is not None:
            position_ids = self._compute_mrope_position_ids(
                seqs, image_grid_thw
            )
            self.pos_encoder.set_position_ids(position_ids)

        # Frontend: embed text + encode/merge vision
        embeds, seqs_layout = self.decoder_frontend(
            seqs,
            seqs_layout,
            state_bag=state_bag,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Decoder: run through transformer layers
        hidden = self.decoder(embeds, seqs_layout, state_bag=state_bag)

        # Project to vocab
        logits = self.final_proj(hidden)

        # Clear position IDs after use
        if self.pos_encoder is not None:
            self.pos_encoder.set_position_ids(None)

        return logits

    def _compute_mrope_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Tensor | None,
    ) -> Tensor | None:
        """Compute 3D M-RoPE position IDs.

        For text-only: all 3 sections use sequential positions.
        For multimodal: vision tokens get spatial (t, h, w) positions.

        Returns:
            position_ids: (B, 3, S) or None for text-only.
        """
        B, S = input_ids.shape
        device = input_ids.device

        if image_grid_thw is None or image_grid_thw.numel() == 0:
            # Text-only: sequential positions for all 3 sections
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            return positions.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, S)

        # Multimodal: compute per-token 3D positions
        position_ids = torch.zeros(B, 3, S, device=device, dtype=torch.long)

        for b in range(B):
            seq = input_ids[b]  # (S,)

            # Find image token spans
            image_mask = seq == self.image_token_id
            text_pos = 0
            img_idx = 0

            # Walk through the sequence assigning positions
            t_pos = torch.zeros(S, device=device, dtype=torch.long)
            h_pos = torch.zeros(S, device=device, dtype=torch.long)
            w_pos = torch.zeros(S, device=device, dtype=torch.long)

            i = 0
            while i < S:
                if not image_mask[i]:
                    # Text token: all 3 sections use same sequential position
                    t_pos[i] = text_pos
                    h_pos[i] = text_pos
                    w_pos[i] = text_pos
                    text_pos += 1
                    i += 1
                else:
                    # Image token span: find contiguous image tokens
                    span_start = i
                    while i < S and image_mask[i]:
                        i += 1
                    span_len = i - span_start

                    if img_idx < image_grid_thw.shape[0]:
                        t_grid = image_grid_thw[img_idx, 0].item()
                        h_grid = image_grid_thw[img_idx, 1].item()
                        w_grid = image_grid_thw[img_idx, 2].item()
                        img_idx += 1

                        # After merger's 2x2 spatial merge, the token count is:
                        # t_grid * (h_grid // merge) * (w_grid // merge)
                        merge = 2  # spatial_merge_size
                        h_merged = h_grid // merge
                        w_merged = w_grid // merge

                        for j in range(span_len):
                            # Map flat index to (t, h, w) in merged grid
                            tokens_per_frame = h_merged * w_merged
                            t_idx = j // tokens_per_frame
                            rem = j % tokens_per_frame
                            h_idx = rem // w_merged
                            w_idx = rem % w_merged

                            pos = span_start + j
                            t_pos[pos] = text_pos + t_idx
                            h_pos[pos] = text_pos + h_idx
                            w_pos[pos] = text_pos + w_idx

                        # Advance text position past the image
                        text_pos += max(t_grid, h_merged, w_merged)
                    else:
                        # Fallback: treat as text
                        for j in range(span_len):
                            pos = span_start + j
                            t_pos[pos] = text_pos
                            h_pos[pos] = text_pos
                            w_pos[pos] = text_pos
                            text_pos += 1

            position_ids[b, 0] = t_pos
            position_ids[b, 1] = h_pos
            position_ids[b, 2] = w_pos

        return position_ids

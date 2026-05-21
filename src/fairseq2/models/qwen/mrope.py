# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal Rotary Position Encoding (M-RoPE) for Qwen 3.6.

M-RoPE splits the rotary encoding dimension (64 dims, from partial_rotary_factor=0.25
of head_dim=256) into 3 sections: [11, 11, 10] frequency pairs = 22+22+20 = 64 dims.
Each section gets independent position IDs:
  - Text tokens: all 3 sections use sequential text positions
  - Vision tokens: section 0 = temporal, section 1 = height, section 2 = width

Design: ``MultimodalRotaryEncoder`` wraps ``ReferenceRotaryEncoder``. Position IDs
are set on the encoder instance before each decoder forward pass, avoiding interface
changes to ``TransformerLMDecoder``.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.nn import BatchLayout
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder


class MultimodalRotaryEncoder(ReferenceRotaryEncoder):
    """RoPE encoder with 3-section multimodal position IDs.

    When ``position_ids`` is set (via :meth:`set_position_ids`), each of the 3
    sections uses its own position sequence. When not set, falls back to
    standard sequential 1D positions (text-only mode).
    """

    mrope_section: Final[list[int]]

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        theta: float = 10_000_000.0,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__(encoding_dim, max_seq_len, theta=theta)

        if mrope_section is None:
            mrope_section = [11, 11, 10]
        self.mrope_section = mrope_section

        # Validate: sum of section pairs * 2 must equal encoding_dim
        total_pairs = sum(mrope_section)
        if total_pairs * 2 != encoding_dim:
            raise ValueError(
                f"Sum of mrope_section ({mrope_section}) = {total_pairs} pairs "
                f"= {total_pairs * 2} dims, but encoding_dim = {encoding_dim}."
            )

        # Mutable state: set before each forward pass for multimodal inputs.
        # Shape: (B, 3, S) — 3 position sequences per batch element.
        self._position_ids: Tensor | None = None

    def set_position_ids(self, position_ids: Tensor | None) -> None:
        """Set 3D position IDs for the next forward pass.

        Args:
            position_ids: (B, 3, S) tensor, or None to use standard 1D positions.
        """
        self._position_ids = position_ids

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if self._position_ids is None:
            # Standard 1D RoPE — delegate to parent.
            return super().forward(seqs, seqs_layout, state_bag=state_bag)

        # Multimodal M-RoPE path.
        return self._mrope_forward(seqs, seqs_layout, state_bag=state_bag)

    def _mrope_forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """Apply M-RoPE with per-section position IDs."""
        position_ids = self._position_ids
        assert position_ids is not None

        # position_ids: (B, 3, S)
        # seqs: (B, S, H, encoding_dim) or (B, S, encoding_dim)
        # We need to apply different cos/sin per section.

        # cos_freqs, sin_freqs are stored as (max_seq_len+1, encoding_dim)
        # Index 0 is padding; actual positions start at index 1.
        # So for position p, use index p+1.

        fp32_seqs = seqs.float()

        # Compute section boundaries in the encoding dimension
        # mrope_section = [11, 11, 10] means 11, 11, 10 frequency PAIRS
        # Each pair covers 2 dims in the cos/sin table layout:
        #   cos_freqs layout: [pair0_cos, pair1_cos, ..., pairN_cos, pair0_cos_repeat, ...]
        # Actually, ReferenceRotaryEncoder stores cos as:
        #   cos_freqs[:, :E//2] = cos(table)
        #   cos_freqs[:, E//2:] = cos(table)
        # And _rotate_half_way splits at E//2.
        # So each "section" in terms of pair indices maps to:
        #   section i occupies pair_indices [sum(mrope_section[:i]), sum(mrope_section[:i+1]))
        #   which maps to cos_freqs[:, pair_start:pair_end] and cos_freqs[:, E//2+pair_start:E//2+pair_end]

        E = self.encoding_dim
        half_E = E // 2

        output_parts = []
        pair_offset = 0

        for sec_idx, num_pairs in enumerate(self.mrope_section):
            # Position IDs for this section: (B, S)
            sec_pos = position_ids[:, sec_idx, :]  # (B, S)
            # Shift by +1 for the padding row in cos/sin tables
            sec_pos_idx = sec_pos + 1  # (B, S)

            # Gather cos/sin for this section's positions
            # cos_freqs: (max_seq_len+1, E)
            # We need: (B, S, num_pairs) from the first half, and same from second half
            cos_sec = self.cos_freqs[sec_pos_idx.long()]  # (B, S, E)
            sin_sec = self.sin_freqs[sec_pos_idx.long()]  # (B, S, E)

            # Extract the relevant pairs for this section
            cos_first = cos_sec[..., pair_offset : pair_offset + num_pairs]  # (B, S, num_pairs)
            cos_second = cos_sec[..., half_E + pair_offset : half_E + pair_offset + num_pairs]
            sin_first = sin_sec[..., pair_offset : pair_offset + num_pairs]
            sin_second = sin_sec[..., half_E + pair_offset : half_E + pair_offset + num_pairs]

            # Corresponding dimensions in seqs
            dim_start_first = pair_offset
            dim_end_first = pair_offset + num_pairs
            dim_start_second = half_E + pair_offset
            dim_end_second = half_E + pair_offset + num_pairs

            # Extract seq slices
            seq_first = fp32_seqs[..., dim_start_first:dim_end_first]
            seq_second = fp32_seqs[..., dim_start_second:dim_end_second]

            # Apply rotation: x * cos + rotate_half(x) * sin
            # For the "half" layout, rotate_half swaps first/second halves with sign flip
            rot_first = seq_first * cos_first + (-seq_second) * sin_first
            rot_second = seq_second * cos_second + seq_first * sin_second

            output_parts.append((dim_start_first, rot_first))
            output_parts.append((dim_start_second, rot_second))

            pair_offset += num_pairs

        # Assemble output
        output = torch.zeros_like(fp32_seqs)
        for start, part in output_parts:
            width = part.shape[-1]
            output[..., start : start + width] = part

        return output.type_as(seqs)

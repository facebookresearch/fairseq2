# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""YaRN-scaled Rotary Position Encoder for OLMO3 long-context models."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from fairseq2.nn.utils.module import unsqueeze


class YaRNRotaryEncoder(ReferenceRotaryEncoder):  # type: ignore[misc]
    """YaRN-scaled Rotary Position Encoder inheriting from ReferenceRotaryEncoder.

    Implements YaRN (Yet another RoPE extensioN) scaling for long-context extension.
    YaRN applies frequency-dependent RoPE scaling:
    - High frequencies (short wavelengths): No scaling (extrapolation)
    - Low frequencies (long wavelengths): Scaled by factor (interpolation)
    - Medium frequencies: Smooth linear ramp transition

    This implementation matches the HuggingFace transformers YaRN implementation
    exactly, as used in OLMO3 long-context models.

    Reference: https://arxiv.org/abs/2309.00071

    Note: Inherits from ReferenceRotaryEncoder which is marked as @final.
    The type: ignore comment suppresses the type checker warning about
    subclassing a final class. This is intentional to maximize code reuse
    while implementing YaRN-specific frequency scaling.
    """

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        theta: float = 500_000.0,
        scale_factor: float,
        original_max_seq_len: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        truncate: bool = True,
        device: Device | None = None,
    ) -> None:
        """Initialize YaRN Rotary Encoder.

        Args:
            encoding_dim: The dimensionality of positional encodings (head_dim).
            max_seq_len: The maximum allowed length for input sequences (extended, e.g., 65536).
            theta: The coefficient of the long-term decay (RoPE base).
            scale_factor: Context extension ratio (e.g., 8.0 for 8K→65K).
            original_max_seq_len: Original max sequence length before extension.
            beta_fast: Boundary for high-frequency extrapolation (default: 32).
            beta_slow: Boundary for low-frequency interpolation (default: 1).
            mscale: Numerator scalar for attention scaling computation.
            mscale_all_dim: Denominator scalar for attention scaling.
            truncate: If True, truncate correction range bounds to integers (default: True).
            device: The device to use for initialization.
        """
        # Initialize parent class
        super().__init__(encoding_dim, max_seq_len, theta=theta, device=device)

        # Store YaRN-specific parameters
        self.scale_factor = scale_factor
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale_param = mscale
        self.mscale_all_dim = mscale_all_dim
        self.truncate = truncate

        # Compute attention scaling factor
        self.attention_scaling = self._compute_attention_scaling()

    def _compute_attention_scaling(self) -> float:
        """Compute attention scaling factor (mscale) for YaRN."""
        def get_mscale(scale: float, mscale_val: float = 1.0) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * mscale_val * math.log(scale) + 1.0

        if self.mscale_param and self.mscale_all_dim:
            return float(
                get_mscale(self.scale_factor, self.mscale_param)
                / get_mscale(self.scale_factor, self.mscale_all_dim)
            )
        else:
            return get_mscale(self.scale_factor, self.mscale_param or 1.0)

    @override
    def reset_non_persistent_buffers(self) -> None:
        """Pre-compute cos/sin tables using YaRN-scaled frequencies."""
        self.cos_freqs[0] = 0.0  # pad
        self.sin_freqs[0] = 0.0  # pad

        dtype = torch.float32
        device = self.cos_freqs.device
        encoding_dim = self.encoding_dim

        # Compute YaRN-scaled inverse frequencies
        inv_freq = self._compute_yarn_inv_freq(device)

        # (E/2) -> (1, E/2)
        inv_freq = inv_freq.unsqueeze(0)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (S, 1)
        steps = steps.unsqueeze(1)

        # (S, 1) x (1, E/2) -> (S, E/2)
        table = steps * inv_freq

        cos = torch.cos(table)
        sin = torch.sin(table)

        # Duplicate for both halves (same as ReferenceRotaryEncoder)
        self.cos_freqs[1:, : encoding_dim // 2] = cos
        self.cos_freqs[1:, encoding_dim // 2 :] = cos

        self.sin_freqs[1:, : encoding_dim // 2] = sin
        self.sin_freqs[1:, encoding_dim // 2 :] = sin

    def _compute_yarn_inv_freq(self, device: Device) -> Tensor:
        """Compute YaRN-scaled inverse frequencies.

        Matches HuggingFace _compute_yarn_parameters exactly.
        """
        dim = self.encoding_dim
        base = self.theta
        factor = self.scale_factor
        original_max_seq_len = self.original_max_seq_len
        beta_fast = self.beta_fast
        beta_slow = self.beta_slow

        # Helper functions (matches HuggingFace)
        def find_correction_dim(num_rot: float, d: int, b: float, max_pos: int) -> float:
            """Inverse dimension formula to find dimension based on rotations."""
            return (d * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(b))

        def find_correction_range(
            low_rot: float, high_rot: float, d: int, b: float, max_pos: int, truncate: bool
        ) -> tuple[float, float]:
            """Find dimension range bounds based on rotations."""
            low = find_correction_dim(low_rot, d, b, max_pos)
            high = find_correction_dim(high_rot, d, b, max_pos)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, d - 1)

        def linear_ramp(min_val: float, max_val: float, d: int, dev: Device) -> Tensor:
            """Create linear ramp from 0 to 1 over dimension range."""
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity
            linear = (torch.arange(d, dtype=torch.float32, device=dev) - min_val) / (max_val - min_val)
            return torch.clamp(linear, 0, 1)

        # Compute base frequencies
        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs  # High freq: no scaling
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)  # Low freq: scaled

        # Find correction range using truncate parameter
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_seq_len, self.truncate)

        # Compute blend factor (ramp from 0 to 1) - directly on the correct device
        ramp = 1 - linear_ramp(low, high, dim // 2, device)

        # Blend: where ramp=0 (high freq) use extrapolation, where ramp=1 (low freq) use interpolation
        inv_freq = inv_freq_interpolation * (1 - ramp) + inv_freq_extrapolation * ramp

        return inv_freq

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """Apply YaRN-scaled rotary encoding with attention scaling."""
        if not self.training and state_bag is not None:
            start_step = state_bag.step_nr
        else:
            start_step = 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        if seqs_layout.packed or seqs_layout.padded:
            indices = seqs_layout.position_indices + 1  # +1 for padding

            if not self.training and state_bag is not None:
                indices = state_bag.step_nr + indices

            # ([N], S, E)
            cos_freqs = self.cos_freqs[indices]
            sin_freqs = self.sin_freqs[indices]
        else:
            batch_width = seqs_layout.width

            if not self.training and state_bag is not None:
                start_step = 1 + state_bag.step_nr
            else:
                start_step = 1

            # (S, E)
            cos_freqs = self.cos_freqs[start_step : start_step + batch_width]
            sin_freqs = self.sin_freqs[start_step : start_step + batch_width]

            # (S, E) -> (1, S, E)
            cos_freqs = cos_freqs.unsqueeze(0)
            sin_freqs = sin_freqs.unsqueeze(0)

        # Apply YaRN attention scaling (this is the key difference from parent)
        cos_freqs = cos_freqs * self.attention_scaling
        sin_freqs = sin_freqs * self.attention_scaling

        if d := seqs.ndim - cos_freqs.ndim:
            cos_freqs = unsqueeze(cos_freqs, dim=-2, count=d)
            sin_freqs = unsqueeze(sin_freqs, dim=-2, count=d)

        fp32_seqs = seqs.float()

        fp32_rotated_seqs = self._rotate_half_way(fp32_seqs)

        fp32_seqs = (fp32_seqs * cos_freqs) + (fp32_rotated_seqs * sin_freqs)

        return fp32_seqs.type_as(seqs)

    @override
    def extra_repr(self) -> str:
        return (
            f"encoding_dim={self.encoding_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}, "
            f"scale_factor={self.scale_factor}, "
            f"attention_scaling={self.attention_scaling:.4f}"
        )

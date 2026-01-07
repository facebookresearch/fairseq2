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

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.olmo.config import YaRNScaleConfig
from fairseq2.nn import BatchLayout, IncrementalStateBag, PositionEncoder
from fairseq2.nn.utils.module import unsqueeze


class YaRNRotaryEncoder(PositionEncoder):
    """YaRN-scaled Rotary Position Encoder for long-context OLMO3 models.
    
    Implements YaRN (Yet another RoPE extensioN) scaling to extend context length
    from 8K to 65K tokens. YaRN applies frequency-dependent scaling where:
    - High frequencies (short wavelengths): No scaling (extrapolation)
    - Low frequencies (long wavelengths): Scale down by factor (interpolation)
    - Medium frequencies: Smooth transition via linear ramp
    
    This implementation matches the HuggingFace transformers YaRN implementation
    exactly, as used in OLMO3 long-context models.
    
    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        theta: float = 500_000.0,
        yarn_config: YaRNScaleConfig,
        device: Device | None = None,
    ) -> None:
        """Initialize YaRN Rotary Encoder.
        
        Args:
            encoding_dim: The dimensionality of positional encodings (head_dim).
            max_seq_len: The maximum allowed length for input sequences (extended length, e.g., 65536).
            theta: The coefficient of the long-term decay (RoPE base).
            yarn_config: YaRN scaling configuration.
            device: The device to use for initialization.
        """
        super().__init__(encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.yarn_config = yarn_config

        # Compute YaRN-scaled inverse frequencies
        inv_freq, attention_scaling = self._compute_yarn_frequencies(
            encoding_dim, theta, yarn_config, device
        )

        # Pre-compute cos/sin tables (same as ReferenceRotaryEncoder)
        cos_freqs = torch.empty(
            (max_seq_len + 1, encoding_dim), device=device, dtype=torch.float32
        )
        sin_freqs = torch.empty(
            (max_seq_len + 1, encoding_dim), device=device, dtype=torch.float32
        )

        self.cos_freqs: Tensor
        self.sin_freqs: Tensor

        self.register_buffer("cos_freqs", cos_freqs, persistent=False)
        self.register_buffer("sin_freqs", sin_freqs, persistent=False)
        
        self.attention_scaling = attention_scaling

        # Store YaRN inv_freq for reset
        self._yarn_inv_freq = inv_freq

        self.reset_parameters()

    def _compute_yarn_frequencies(
        self,
        dim: int,
        base: float,
        config: YaRNScaleConfig,
        device: Device | None,
    ) -> tuple[Tensor, float]:
        """Compute YaRN-scaled inverse frequencies.
        
        This matches the HuggingFace _compute_yarn_parameters implementation exactly.
        
        Args:
            dim: Head dimension
            base: RoPE theta (base frequency)
            config: YaRN configuration
            device: Device for tensors
            
        Returns:
            Tuple of (yarn_scaled_inv_freq, attention_scaling_factor)
        """
        factor = config.scale_factor
        original_max_seq_len = config.original_max_seq_len
        beta_fast = 32  # Default from HuggingFace
        beta_slow = 1   # Default from HuggingFace

        # Compute attention scaling factor (mscale)
        def get_mscale(scale: float, mscale: float = 1.0) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale = config.mscale
        mscale_all_dim = config.mscale_all_dim

        if mscale and mscale_all_dim:
            attention_scaling = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_scaling = get_mscale(factor, mscale or 1.0)

        # Helper functions from HuggingFace
        def find_correction_dim(
            num_rotations: float, dim: int, base: float, max_position_embeddings: int
        ) -> float:
            """Inverse dimension formula to find the dimension based on the number of rotations."""
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
                2 * math.log(base)
            )

        def find_correction_range(
            low_rot: float,
            high_rot: float,
            dim: int,
            base: float,
            max_position_embeddings: int,
            truncate: bool,
        ) -> tuple[float, float]:
            """Find dimension range bounds based on rotations."""
            low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
            high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_val: float, max_val: float, dim: int) -> Tensor:
            """Create linear ramp from 0 to 1 over dimension range."""
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
                max_val - min_val
            )
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        # Compute base inverse frequencies (before YaRN scaling)
        pos_freqs = base ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs  # High freq (no scaling)
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)  # Low freq (scaled)

        # Find correction range
        truncate = True  # Default in HuggingFace
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_seq_len, truncate
        )

        # Get n-dimensional rotational scaling corrected for extrapolation
        # This creates a smooth transition from extrapolation to interpolation
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)
        if device is not None:
            inv_freq_extrapolation_factor = inv_freq_extrapolation_factor.to(device)

        # Blend between extrapolation and interpolation based on ramp
        # - Where ramp=0 (high freq): use extrapolation (no scaling)
        # - Where ramp=1 (low freq): use interpolation (scaled by factor)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

        return inv_freq, attention_scaling

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Pre-compute cos/sin tables using YaRN-scaled frequencies."""
        self.cos_freqs[0] = 0.0  # pad
        self.sin_freqs[0] = 0.0  # pad

        dtype = torch.float32
        device = self.cos_freqs.device
        encoding_dim = self.encoding_dim

        # Use YaRN-scaled inverse frequencies
        inv_freq = self._yarn_inv_freq
        if inv_freq.device != device:
            inv_freq = inv_freq.to(device)

        # (E/2) -> (1, E/2)
        inv_freq = inv_freq.unsqueeze(0)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (S, 1)
        steps = steps.unsqueeze(1)

        # (S, 1) x (1, E/2) -> (S, E/2)
        # Note: This is different from ReferenceRotaryEncoder which uses theta ** (-2.0 * indices / dim)
        # YaRN pre-computed inv_freq, so we just multiply by steps
        table = steps * inv_freq

        cos = torch.cos(table)
        sin = torch.sin(table)

        # Duplicate for both halves (same as ReferenceRotaryEncoder)
        self.cos_freqs[1:, : encoding_dim // 2] = cos
        self.cos_freqs[1:, encoding_dim // 2 :] = cos

        self.sin_freqs[1:, : encoding_dim // 2] = sin
        self.sin_freqs[1:, encoding_dim // 2 :] = sin

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """Apply YaRN-scaled rotary encoding.
        
        Same as ReferenceRotaryEncoder but applies attention_scaling to cos/sin.
        """
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

        # Apply YaRN attention scaling
        cos_freqs = cos_freqs * self.attention_scaling
        sin_freqs = sin_freqs * self.attention_scaling

        if d := seqs.ndim - cos_freqs.ndim:
            cos_freqs = unsqueeze(cos_freqs, dim=-2, count=d)
            sin_freqs = unsqueeze(sin_freqs, dim=-2, count=d)

        fp32_seqs = seqs.float()

        fp32_rotated_seqs = self._rotate_half_way(fp32_seqs)

        fp32_seqs = (fp32_seqs * cos_freqs) + (fp32_rotated_seqs * sin_freqs)

        return fp32_seqs.type_as(seqs)

    def _rotate_half_way(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input (HuggingFace-style)."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"encoding_dim={self.encoding_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}, "
            f"yarn_scale_factor={self.yarn_config.scale_factor}, "
            f"attention_scaling={self.attention_scaling:.4f}"
        )

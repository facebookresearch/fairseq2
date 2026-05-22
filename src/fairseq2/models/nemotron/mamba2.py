# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba2 Selective State Space Model for NemotronH.

This module implements the Mamba2 SSM (Structured State Space Duality) as used
in NVIDIA's Nemotron-H architecture. The key components are:

1. in_proj: Projects input to gate, x_BC (conv input), and dt
2. conv1d: Depthwise causal convolution on x_BC
3. Selective scan: The core SSM via mamba_chunk_scan_combined (chunked SSD)
4. Gated RMSNorm: Normalizes scan output and gates with the gate projection
5. out_proj: Projects back to model dimension

The module supports two modes:
- Training: Full sequence processing via mamba_chunk_scan_combined
- Generation: Incremental decoding with conv_state and ssm_state caches
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from fairseq2.nn import IncrementalState, IncrementalStateBag

# Try to import CUDA-accelerated Mamba2 ops
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False


@dataclass
class NemotronHMamba2State(IncrementalState):
    """Incremental state for Mamba2 generation.

    Attributes:
        conv_state: The shift register for the causal conv1d.
            Shape: [batch_size, conv_dim, kernel_size]
        ssm_state: The recurrent SSM state per head.
            Shape: [batch_size, num_heads, head_dim, state_size]
    """

    conv_state: Tensor
    ssm_state: Tensor

    def reorder(self, new_order: Tensor) -> None:
        self.conv_state = self.conv_state.index_select(0, new_order)
        self.ssm_state = self.ssm_state.index_select(0, new_order)

    def size_bytes(self) -> int:
        return (
            self.conv_state.nelement() * self.conv_state.element_size()
            + self.ssm_state.nelement() * self.ssm_state.element_size()
        )

    def capacity_bytes(self) -> int:
        return self.size_bytes()


class RMSNormGated(nn.Module):
    """Gated RMS Normalization as used in Mamba2 (Zamba2RMSNormGated).

    Applies RMS normalization to the first half of the input and gates it
    with the SiLU-activated second half:
        output = rms_norm(hidden_states) * silu(gate)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @override
    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        """Apply gated RMS normalization.

        Args:
            hidden_states: Input to normalize. Shape: [..., hidden_size]
            gate: Gate values. Shape: [..., hidden_size]

        Returns:
            Normalized and gated output. Shape: [..., hidden_size]
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype) * self.weight

        # Gate with SiLU activation
        hidden_states = hidden_states * F.silu(gate.to(input_dtype))
        return hidden_states


@final
class NemotronHMamba2Mixer(nn.Module):
    """Mamba2 Selective State Space Model mixer for NemotronH.

    This implements the full Mamba2 forward pass:
    1. in_proj: [B,L,D] -> [B,L, gate+x_BC+dt]
    2. Split into gate, x_BC, dt
    3. conv1d on x_BC (depthwise, kernel=4)
    4. Split conv output into x, B, C
    5. Selective scan (chunked SSD)
    6. Gated RMSNorm(scan_output, gate)
    7. out_proj: [B,L,intermediate] -> [B,L,D]
    """

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int = 64,
        head_dim: int = 64,
        state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 128,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        use_conv_bias: bool = True,
        proj_bias: bool = False,
        eps: float = 1e-5,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Derived dimensions
        self.intermediate_size = num_heads * head_dim  # 4096
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size  # 6144
        self.projection_size = self.intermediate_size + self.conv_dim + num_heads  # 10304

        # Input projection: D -> gate + x_BC + dt
        self.in_proj = nn.Linear(model_dim, self.projection_size, bias=proj_bias)

        # Depthwise causal conv1d on x_BC
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=conv_kernel,
            groups=self.conv_dim,  # depthwise
            padding=conv_kernel - 1,  # causal padding
            bias=use_conv_bias,
        )

        # Learnable SSM parameters
        # A_log: log-space decay rates per head
        self.A_log = nn.Parameter(torch.empty(num_heads))
        # D: skip connection per head
        self.D = nn.Parameter(torch.empty(num_heads))
        # dt_bias: bias for the time-step projection
        self.dt_bias = nn.Parameter(torch.empty(num_heads))

        # Gated RMSNorm (Zamba2RMSNormGated)
        self.norm = RMSNormGated(self.intermediate_size, eps=eps)

        # Output projection: intermediate -> D
        self.out_proj = nn.Linear(self.intermediate_size, model_dim, bias=proj_bias)

        # Initialize parameters
        self._init_parameters(time_step_min, time_step_max)

    def _init_parameters(
        self, time_step_min: float, time_step_max: float
    ) -> None:
        """Initialize Mamba2 parameters following the HF reference."""
        # A_log initialization: uniform in [1, num_heads]
        A = torch.arange(1, self.num_heads + 1, dtype=torch.float32)
        self.A_log.data = torch.log(A)

        # D initialization: ones
        nn.init.ones_(self.D)

        # dt_bias initialization: log-uniform in [time_step_min, time_step_max]
        dt = torch.exp(
            torch.rand(self.num_heads)
            * (math.log(time_step_max) - math.log(time_step_min))
            + math.log(time_step_min)
        )
        # Inverse softplus so that softplus(dt_bias) gives the desired dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data = inv_dt

    @override
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Forward pass of Mamba2 mixer.

        Args:
            hidden_states: Input tensor. Shape: [batch, seq_len, model_dim]
            attention_mask: Optional attention mask (used for padding).
            state_bag: Incremental state bag for generation.

        Returns:
            Output tensor. Shape: [batch, seq_len, model_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Check for incremental decoding (generation mode)
        state: NemotronHMamba2State | None = None
        if state_bag is not None:
            state = state_bag.maybe_get_state(self, NemotronHMamba2State)

        if state is not None:
            # Incremental decoding: process one token at a time
            return self._forward_incremental(hidden_states, state)

        # Full sequence forward
        output = self._forward_training(hidden_states, attention_mask)

        # Initialize state for future incremental decoding
        if state_bag is not None:
            conv_state = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.state_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            state = NemotronHMamba2State(conv_state=conv_state, ssm_state=ssm_state)
            state_bag.set_state(self, state)

        return output

    def _forward_training(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Full sequence forward for training.

        Uses mamba_chunk_scan_combined when CUDA kernels are available,
        otherwise falls back to a pure PyTorch implementation.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Input projection
        projected = self.in_proj(hidden_states)  # [B, L, projection_size]

        # 2. Split into gate, x_BC, dt
        gate, x_BC, dt = projected.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # 3. Causal conv1d on x_BC
        x_BC = self._apply_conv1d(x_BC)

        # 4. Split conv output into hidden_states (x), B, C
        x, B, C = x_BC.split(
            [
                self.intermediate_size,
                self.n_groups * self.state_size,
                self.n_groups * self.state_size,
            ],
            dim=-1,
        )

        # Reshape for SSM
        # x: [B, L, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # B: [B, L, n_groups, state_size]
        B = B.view(batch_size, seq_len, self.n_groups, self.state_size)
        # C: [B, L, n_groups, state_size]
        C = C.view(batch_size, seq_len, self.n_groups, self.state_size)
        # dt: [B, L, num_heads]
        # Already the right shape

        # 5. Selective scan
        A = -torch.exp(self.A_log.float())  # [num_heads]

        if HAS_MAMBA_SSM:
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,  # No gating inside scan
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
            )
            # y: [B, L, num_heads, head_dim]
        else:
            y = self._selective_scan_pytorch(x, dt, A, B, C)

        # Reshape back to [B, L, intermediate_size]
        y = y.view(batch_size, seq_len, self.intermediate_size)

        # 6. Gated RMSNorm
        y = self.norm(y, gate)

        # 7. Output projection
        output = self.out_proj(y)

        return output

    def _apply_conv1d(self, x_BC: Tensor) -> Tensor:
        """Apply causal conv1d to x_BC.

        Args:
            x_BC: Input tensor. Shape: [B, L, conv_dim]

        Returns:
            Conv output with SiLU activation. Shape: [B, L, conv_dim]
        """
        if HAS_CAUSAL_CONV1D:
            # Use optimized causal conv1d
            x_BC = x_BC.transpose(1, 2)  # [B, conv_dim, L]
            x_BC = causal_conv1d_fn(
                x=x_BC,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            )
            x_BC = x_BC.transpose(1, 2)  # [B, L, conv_dim]
        else:
            # PyTorch fallback
            x_BC = x_BC.transpose(1, 2)  # [B, conv_dim, L]
            x_BC = self.conv1d(x_BC)[..., :x_BC.shape[-1]]  # Causal: trim future
            x_BC = F.silu(x_BC)
            x_BC = x_BC.transpose(1, 2)  # [B, L, conv_dim]

        return x_BC

    def _selective_scan_pytorch(
        self,
        x: Tensor,  # [B, L, H, D]
        dt: Tensor,  # [B, L, H]
        A: Tensor,  # [H]
        B: Tensor,  # [B, L, G, N]
        C: Tensor,  # [B, L, G, N]
    ) -> Tensor:
        """Pure PyTorch fallback for selective scan.

        This is a simple sequential scan (not chunked) for correctness reference.
        Much slower than the CUDA kernel but numerically equivalent.

        Returns:
            Output tensor. Shape: [B, L, H, D]
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        state_size = B.shape[-1]
        n_groups = B.shape[2]
        heads_per_group = num_heads // n_groups

        # Apply softplus to dt and add bias
        dt = F.softplus(dt + self.dt_bias)  # [B, L, H]

        # Expand B, C from groups to heads
        # B: [B, L, G, N] -> [B, L, H, N]
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)

        # Initialize SSM state
        ssm_state = torch.zeros(
            batch_size, num_heads, head_dim, state_size,
            device=x.device, dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            # Discretize: dA = exp(A * dt)
            dt_t = dt[:, t, :]  # [B, H]
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # [1, H] * [B, H] -> [B, H]
            dA = dA.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            # dB = dt * B
            x_t = x[:, t, :, :]  # [B, H, D]
            B_t = B[:, t, :, :]  # [B, H, N]
            dt_t_expanded = dt_t.unsqueeze(-1)  # [B, H, 1]

            # State update: s = dA * s + dB * x
            dBx = torch.einsum("bhd,bhn->bhdn", x_t * dt_t_expanded, B_t)
            ssm_state = dA * ssm_state + dBx

            # Output: y = C @ s + D * x
            C_t = C[:, t, :, :]  # [B, H, N]
            y_t = torch.einsum("bhdn,bhn->bhd", ssm_state, C_t)
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_t  # skip connection

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, L, H, D]

    def _forward_incremental(
        self,
        hidden_states: Tensor,
        state: NemotronHMamba2State,
    ) -> Tensor:
        """Incremental forward for generation (single token).

        Args:
            hidden_states: Input tensor. Shape: [B, 1, model_dim]
            state: Cached conv_state and ssm_state.

        Returns:
            Output tensor. Shape: [B, 1, model_dim]
        """
        batch_size = hidden_states.shape[0]

        # 1. Input projection
        projected = self.in_proj(hidden_states.squeeze(1))  # [B, projection_size]

        # 2. Split
        gate, x_BC, dt = projected.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # 3. Causal conv1d update (single step)
        if HAS_CAUSAL_CONV1D:
            x_BC = causal_conv1d_update(
                x=x_BC,
                conv_state=state.conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            )
        else:
            # PyTorch fallback: shift register
            state.conv_state.copy_(torch.roll(state.conv_state, shifts=-1, dims=-1))
            state.conv_state[:, :, -1] = x_BC
            x_BC = torch.sum(
                state.conv_state * self.conv1d.weight.squeeze(1).flip(-1), dim=-1
            )
            if self.conv1d.bias is not None:
                x_BC = x_BC + self.conv1d.bias
            x_BC = F.silu(x_BC)

        # 4. Split conv output
        x, B_val, C_val = x_BC.split(
            [
                self.intermediate_size,
                self.n_groups * self.state_size,
                self.n_groups * self.state_size,
            ],
            dim=-1,
        )

        # Reshape
        x = x.view(batch_size, self.num_heads, self.head_dim)
        B_val = B_val.view(batch_size, self.n_groups, self.state_size)
        C_val = C_val.view(batch_size, self.n_groups, self.state_size)

        # Expand B, C from groups to heads
        heads_per_group = self.num_heads // self.n_groups
        B_val = B_val.repeat_interleave(heads_per_group, dim=1)
        C_val = C_val.repeat_interleave(heads_per_group, dim=1)

        # 5. SSM state update
        A = -torch.exp(self.A_log.float())
        dt_val = F.softplus(dt + self.dt_bias)  # [B, H]

        if HAS_MAMBA_SSM:
            y = selective_state_update(
                state.ssm_state,
                x,
                dt_val,
                A,
                B_val,
                C_val,
                D=self.D,
                z=None,
                dt_softplus=False,  # Already applied softplus
            )
        else:
            # PyTorch fallback
            dA = torch.exp(A.unsqueeze(0) * dt_val)  # [1, H] * [B, H] -> [B, H]
            dA = dA.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            dBx = torch.einsum(
                "bhd,bhn->bhdn", x * dt_val.unsqueeze(-1), B_val
            )
            state.ssm_state.copy_(dA * state.ssm_state + dBx)

            y = torch.einsum("bhdn,bhn->bhd", state.ssm_state, C_val)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x

        # Reshape
        y = y.view(batch_size, self.intermediate_size)

        # 6. Gated RMSNorm
        y = self.norm(y, gate)

        # 7. Output projection
        output = self.out_proj(y)

        return output.unsqueeze(1)  # [B, 1, model_dim]

    @override
    def extra_repr(self) -> str:
        return (
            f"model_dim={self.model_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, state_size={self.state_size}, "
            f"n_groups={self.n_groups}, conv_kernel={self.conv_kernel}, "
            f"layer_idx={self.layer_idx}"
        )

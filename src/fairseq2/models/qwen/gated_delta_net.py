# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet linear attention module for Qwen 3.5.

Reference: HuggingFace ``modeling_qwen3_5.py`` lines 445-620.
"""

from __future__ import annotations

import logging
from typing import Callable, Final, final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq2.nn import (
    IncrementalState,
    IncrementalStateBag,
    Linear,
    RMSNorm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional fast-path kernels
# ---------------------------------------------------------------------------

try:
    from causal_conv1d import causal_conv1d_update as _causal_conv1d_update

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False
    logger.warning(
        "causal_conv1d not found; GatedDeltaNet will use a slower PyTorch fallback "
        "for incremental decoding. Install with: pip install causal-conv1d"
    )

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _chunk_gated_delta_rule,
    )
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule as _fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False
    logger.warning(
        "flash-linear-attention (fla) not found; GatedDeltaNet will use slower "
        "pure-PyTorch chunk/recurrent kernels. Install with: pip install flash-linear-attention"
    )


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2-normalize along ``dim``.

    Reference: ``modeling_qwen3_5.py`` lines 317-320.
    """
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# PyTorch fallback kernels (no external dependencies)
# ---------------------------------------------------------------------------


def torch_causal_conv1d_update(
    hidden_states: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """Single-step causal conv1d for incremental decoding.

    Reference: ``modeling_qwen3_5.py`` lines 299-314.

    :param hidden_states: ``(B, D, L)`` — typically ``L=1`` during decode.
    :param conv_state: ``(B, D, kernel-1)`` — updated in-place.
    :param weight: ``(D, kernel)`` — depthwise conv weights.
    :param bias: ``(D,)`` or ``None``.
    :param activation: ``"silu"`` or ``None``.
    :returns: ``(B, D, L)`` convolved output.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])

    out = F.conv1d(
        hidden_states_new,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=hidden_size,
    )
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Chunked gated delta rule for prefill (pure PyTorch).

    Reference: ``modeling_qwen3_5.py`` lines 323-400.

    :param query: ``(B, S, H, K)``
    :param key: ``(B, S, H, K)``
    :param value: ``(B, S, H, V)``
    :param g: ``(B, S, H)`` — forget gate (log-space).
    :param beta: ``(B, S, H)`` — write gate.
    :returns: ``(output, final_state)``
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_seq_len = seq_len + pad_size

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    num_chunks = total_seq_len // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
            mask, 0
        )
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    final_state: Tensor | None = last_state if output_final_state else None

    core_out = core_out.reshape(
        core_out.shape[0], core_out.shape[1], -1, core_out.shape[-1]
    )
    core_out = core_out[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, final_state


def torch_recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Step-by-step recurrent gated delta rule for decode (pure PyTorch).

    Reference: ``modeling_qwen3_5.py`` lines 403-442.

    :param query: ``(B, S, H, K)`` — typically ``S=1`` during decode.
    :param key: ``(B, S, H, K)``
    :param value: ``(B, S, H, V)``
    :param g: ``(B, S, H)`` — forget gate (log-space).
    :param beta: ``(B, S, H)`` — write gate.
    :returns: ``(output, final_state)``
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_out = torch.zeros(batch_size, num_heads, seq_len, v_head_dim).to(value)
    last_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_state = last_state * g_t
        kv_mem = (last_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, i] = (last_state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state: Tensor | None = last_state if output_final_state else None

    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, final_state


# ---------------------------------------------------------------------------
# Incremental state
# ---------------------------------------------------------------------------


@final
class GatedDeltaNetState(IncrementalState):
    """Holds conv and recurrent state for :class:`GatedDeltaNet` during
    incremental decoding."""

    conv_state: Tensor
    """``(B, conv_dim, kernel_size - 1)``"""

    recurrent_state: Tensor
    """``(B, num_v_heads, head_k_dim, head_v_dim)``"""

    def __init__(self, conv_state: Tensor, recurrent_state: Tensor) -> None:
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state

    def reorder(self, new_order: Tensor) -> None:
        self.conv_state = self.conv_state.index_select(0, new_order)
        self.recurrent_state = self.recurrent_state.index_select(0, new_order)

    def size_bytes(self) -> int:
        return self.capacity_bytes()

    def capacity_bytes(self) -> int:
        c = self.conv_state.numel() * self.conv_state.element_size()
        r = self.recurrent_state.numel() * self.recurrent_state.element_size()
        return c + r


# ---------------------------------------------------------------------------
# RMSNormGated — norm-before-gate with silu
# ---------------------------------------------------------------------------


class RMSNormGated(nn.Module):
    """``RMSNorm(x) * silu(gate)``

    Internal norm inside GatedDeltaNet.  Uses the standard ``weight=ones``
    formula (NOT the ``1+weight`` variant used by the outer layer norms).

    Reference: ``modeling_qwen3_5.py`` lines 264-279.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.inner_norm = RMSNorm(dim, bias=False, eps=eps)

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        hidden_states = self.inner_norm(hidden_states)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(gate.dtype)


# ---------------------------------------------------------------------------
# GatedDeltaNet module
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention module for Qwen 3.5.

    Replaces standard multi-head attention in 75% of Qwen 3.5 layers.
    Uses causal convolution followed by a gated delta rule recurrence.

    Reference: ``modeling_qwen3_5.py`` ``Qwen3_5GatedDeltaNet`` lines 445-620.
    """

    hidden_size: Final[int]
    num_k_heads: Final[int]
    num_v_heads: Final[int]
    head_k_dim: Final[int]
    head_v_dim: Final[int]
    key_dim: Final[int]
    value_dim: Final[int]
    conv_dim: Final[int]
    conv_kernel_size: Final[int]

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size

        # Input projections — fairseq2 Linear wrappers.
        self.in_proj_qkv = Linear(
            hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = Linear(hidden_size, num_v_heads, bias=False)

        # Depthwise causal convolution (no fairseq2 wrapper exists).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        # Learnable gating parameters (no fairseq2 wrapper for raw params).
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        A = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Output norm (silu-gated, wraps fairseq2 RMSNorm) and projection.
        self.norm = RMSNormGated(head_v_dim, eps=eps)
        self.out_proj = Linear(self.value_dim, hidden_size, bias=False)

        # Select fast-path kernels when available, else pure-PyTorch fallbacks.
        self._conv1d_update_fn: Callable[..., Tensor] = (
            _causal_conv1d_update if _HAS_CAUSAL_CONV1D else torch_causal_conv1d_update
        )
        self._chunk_fn: Callable[..., tuple[Tensor, Tensor | None]] = (
            _chunk_gated_delta_rule if _HAS_FLA else torch_chunk_gated_delta_rule
        )
        self._recurrent_fn: Callable[..., tuple[Tensor, Tensor | None]] = (
            _fused_recurrent_gated_delta_rule
            if _HAS_FLA
            else torch_recurrent_gated_delta_rule
        )

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs: ``(B, S, D)``
        :param padding_mask: Optional ``(B, S)`` boolean mask (1 = valid).
        :param state_bag: Incremental state bag for generation.
        :returns: ``(B, S, D)``
        """
        if padding_mask is not None and padding_mask.shape[1] > 1:
            seqs = (seqs * padding_mask[:, :, None]).to(seqs.dtype)

        batch_size, seq_len, _ = seqs.shape

        state: GatedDeltaNetState | None = None
        if state_bag is not None:
            state = state_bag.maybe_get_state(self, GatedDeltaNetState)

        use_cache = state is not None and seq_len == 1

        # -- Input projections --
        mixed_qkv = self.in_proj_qkv(seqs).transpose(1, 2)  # (B, conv_dim, S)
        z = self.in_proj_z(seqs).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(seqs)
        a = self.in_proj_a(seqs)

        # -- Causal convolution --
        conv_state: Tensor | None = None

        if use_cache:
            assert state is not None
            mixed_qkv = self._conv1d_update_fn(
                mixed_qkv,
                state.conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                "silu",
            )
        else:
            if state_bag is not None:
                conv_state = F.pad(
                    mixed_qkv,
                    (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                )

            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, S, conv_dim)

        # -- Split QKV --
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # -- Compute gates --
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # -- GQA expansion --
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)

        # -- Delta rule core --
        if use_cache:
            assert state is not None
            core_out, last_state = self._recurrent_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=state.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, last_state = self._chunk_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=state_bag is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # -- Update incremental state --
        if state_bag is not None:
            if state is None:
                assert conv_state is not None and last_state is not None
                state_bag.set_state(self, GatedDeltaNetState(conv_state, last_state))
            else:
                assert last_state is not None
                state.recurrent_state = last_state

        # -- Output norm (silu-gated) + projection --
        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z)
        core_out = core_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_out)

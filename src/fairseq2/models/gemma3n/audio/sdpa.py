# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.nn import BatchLayout
from fairseq2.nn.projection import Linear


@final
class Gemma3nConformerSDPA(Module):
    """Chunked local attention for Gemma3n audio conformer.

    Not a pluggable SDPA backend — conformer's block-based chunking,
    sinusoidal relative positions, pre-softmax softcap, and combined
    validity masking are incompatible with standard SDPA interfaces.
    """

    model_dim: int
    num_heads: int
    head_dim: int
    chunk_size: int
    max_past_horizon: int
    max_future_horizon: int
    context_size: int
    pos_proj: Linear
    per_dim_scale: Parameter

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_pos: int,
        max_right_rel_pos: int,
        chunk_size: int,
        left_context: int,
        right_context: int,
        logit_cap: float,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality (1536).
        :param num_heads: Number of attention heads (8).
        :param max_left_rel_pos: Maximum left relative position.
        :param max_right_rel_pos: Maximum right relative position.
        :param chunk_size: Chunk size for local attention (12).
        :param left_context: Left context size in tokens (13).
        :param right_context: Right context size in tokens (0).
        :param logit_cap: Logit softcapping value (50.0).
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` "
                f"({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.chunk_size = chunk_size
        self.max_past_horizon = max(0, left_context - 1)
        self.max_future_horizon = right_context
        self.context_size = (
            chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.logit_cap = logit_cap

        # Sinusoidal position projection (replaces Shaw embedding)
        self.pos_proj = Linear(
            model_dim,
            num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Per-dimension scaling with zeros init (apply via softplus)
        self.per_dim_scale = Parameter(
            torch.zeros(self.head_dim, device=device, dtype=dtype)
        )

        # q_scale = head_dim^(-0.5) / softplus(0)
        q_scale = self.head_dim**-0.5 / F.softplus(torch.tensor(0.0))
        self.register_buffer(
            "q_scale", q_scale.clone().detach(), persistent=False
        )

        # Sinusoidal inverse timescales
        num_timescales = model_dim // 2
        log_timescale_increment = math.log(1.0e4) / max(
            num_timescales - 1, 1
        )
        inv_timescales = torch.exp(
            torch.arange(num_timescales, dtype=torch.float32)
            * -log_timescale_increment
        )
        self.register_buffer(
            "inv_timescales",
            inv_timescales.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        # Precomputed local causal mask [chunk_size, context_size]
        local_mask = self._create_local_causal_valid_mask()
        self.register_buffer(
            "local_causal_valid_mask", local_mask, persistent=False
        )

        # Softcap value
        self.register_buffer(
            "softcap",
            torch.tensor(logit_cap, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        :param q: Queries. *Shape:* :math:`(N,S,H,K)`.
        :param k: Keys. *Shape:* :math:`(N,S,H,K)`.
        :param v: Values. *Shape:* :math:`(N,S,H,V)`.
        :param mask: Where True=masked (invalid). *Shape:* :math:`(N,T)`.
        :returns: Attention output and optional weights.
        """
        if q_layout.packed or k_layout.packed:
            raise NotSupportedError(
                "Gemma3n conformer SDPA does not support packed batches."
            )

        batch_size, q_time, num_heads, head_dim = q.shape

        # Per-dim scaling with softplus
        per_dim_scale_sp = F.softplus(self.per_dim_scale)
        q = q * (self.q_scale * per_dim_scale_sp)

        # Convert to blocks
        query_blocks = self._convert_to_block(q)
        key_blocks = self._extract_block_context(k)
        value_blocks = self._extract_block_context(v)
        num_query_blocks = query_blocks.shape[1]

        # Build validity mask — always needed because
        # _extract_block_context pads with zeros that must be masked.
        if mask is None:
            mask = torch.zeros(
                batch_size, q_time, dtype=torch.bool, device=q.device
            )
        original_valid = ~mask
        extracted_valid = self._extract_block_context(original_valid)
        # [B, 1, U, 1, C]
        validity_condition = (
            extracted_valid.unsqueeze(1).unsqueeze(-2)
        )

        # Local causal mask [1, 1, 1, W, C]
        causal_condition = (
            self.local_causal_valid_mask
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Compute logits with relative position embeddings
        logits = self._compute_relative_logits(
            query_blocks, key_blocks, num_query_blocks
        )

        # Softcap on logits (pre-softmax)
        softcap_val = self.softcap.to(logits.device)
        logits = torch.tanh(logits / softcap_val) * softcap_val

        # Combined mask: validity AND causality
        final_condition = torch.logical_and(
            validity_condition,
            causal_condition.to(validity_condition.device),
        )

        logits = torch.where(
            final_condition,
            logits,
            torch.finfo(logits.dtype).min,
        )

        # Softmax + weighted sum
        probs = torch.softmax(
            logits, dim=-1, dtype=torch.float32
        ).to(dtype=v.dtype)

        # Context vectors via batched matmul
        # probs: [B, N, U, W, C], values: [B, U, C, N, H]
        b, n, u, w, c = probs.shape
        h = value_blocks.shape[-1]
        prob_bun = probs.permute(0, 2, 1, 3, 4).reshape(-1, w, c)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c, h)
        result = torch.bmm(prob_bun, v_bun)
        context = (
            result.reshape(b, u, n, w, h)
            .permute(0, 1, 3, 2, 4)
            .reshape(b, u * w, n, h)
        )
        # Trim padding to original sequence length
        context = context[:, :q_time]

        return context, None

    if TYPE_CHECKING:
        __call__ = forward

    def _compute_relative_logits(
        self,
        query_blocks: Tensor,
        key_blocks: Tensor,
        num_query_blocks: int,
    ) -> Tensor:
        """Compute attention logits with sinusoidal relative positions.

        :param query_blocks: *Shape:* :math:`(B,U,W,N,K)`.
        :param key_blocks: *Shape:* :math:`(B,U,C,N,K)`.
        :returns: Logits. *Shape:* :math:`(B,N,U,W,C)`.
        """
        batch_size = query_blocks.shape[0]
        _, _, _, num_heads, head_dim = query_blocks.shape
        _, _, key_context_size, _, _ = key_blocks.shape

        # Sinusoidal position embeddings
        pos_indices = torch.arange(
            self.max_past_horizon,
            -self.max_future_horizon - 1,
            -1,
            device=query_blocks.device,
        ).unsqueeze(0)
        f_span = pos_indices.shape[1]

        sin_emb = self._get_timing_signal(
            pos_indices, dtype=query_blocks.dtype
        )
        projected = self.pos_proj(sin_emb)
        sin_emb_heads = projected.reshape(
            1, f_span, num_heads, head_dim
        ).squeeze(0)

        # term_ac: content-content interaction
        # [B, N, U, W, K] @ [B, N, U, K, C] -> [B, N, U, W, C]
        queries_p = query_blocks.permute(0, 3, 1, 2, 4)
        keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
        term_ac = torch.matmul(queries_p, keys_p_t)

        # term_bd: content-position interaction
        # [B, N, U*W, K] @ [N, K, F] -> [B, N, U*W, F]
        q_permuted = queries_p
        s_permuted = sin_emb_heads.permute(1, 2, 0)
        q_reshaped = q_permuted.reshape(
            batch_size,
            num_heads,
            num_query_blocks * self.chunk_size,
            head_dim,
        )
        term_bd_flat = torch.matmul(q_reshaped, s_permuted)
        term_bd = term_bd_flat.reshape(
            batch_size, num_heads, num_query_blocks, self.chunk_size, f_span
        )

        # Relative shift
        term_bd_shifted = self._relative_shift(
            term_bd,
            batch_size,
            num_heads,
            num_query_blocks,
            self.chunk_size,
            key_context_size,
            f_span,
        )

        return term_ac + term_bd_shifted

    def _get_timing_signal(
        self, position: Tensor, dtype: torch.dtype
    ) -> Tensor:
        """Compute sinusoidal timing signal.

        :param position: Position indices. *Shape:* :math:`(1,F)`.
        :returns: Timing signal. *Shape:* :math:`(1,F,D)`.
        """
        position_f = position.float().unsqueeze(-1)
        inv_ts = self.inv_timescales.to(
            device=position.device, dtype=torch.float32
        )
        scaled_time = position_f * inv_ts
        timing_signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1
        )
        return timing_signal.to(dtype)

    @staticmethod
    def _relative_shift(
        term_bd: Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        f_span: int,
    ) -> Tensor:
        """Apply relative shift to align position embeddings with keys.

        :param term_bd: *Shape:* :math:`(B,N,U,W,F)`.
        :returns: Shifted tensor. *Shape:* :math:`(B,N,U,W,C)`.
        """
        pad_amount = (key_context_size + 1) - f_span
        term_bd_padded = F.pad(term_bd, (0, pad_amount))
        term_bd_reshaped = term_bd_padded.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size * (key_context_size + 1),
        )
        term_bd_sliced = term_bd_reshaped[
            :, :, :, : query_block_size * key_context_size
        ]
        return term_bd_sliced.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        )

    def _pad_dim1(
        self, x: Tensor, pad_left: int, pad_right: int
    ) -> Tensor:
        """Zero-pad tensor along dimension 1."""
        batch = x.shape[0]
        tail_shape = x.shape[2:]
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        return torch.cat([left, x, right], dim=1)

    def _convert_to_block(self, x: Tensor) -> Tensor:
        """Split sequence into non-overlapping blocks.

        :param x: *Shape:* :math:`(B,T,...)`.
        :returns: *Shape:* :math:`(B,U,W,...)` where U=ceil(T/W).
        """
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        padding_len = num_blocks * self.chunk_size - t
        if padding_len > 0:
            x = self._pad_dim1(x, 0, padding_len)

        new_shape = (b, num_blocks, self.chunk_size) + shape[2:]
        return x.reshape(new_shape).contiguous()

    def _extract_block_context(self, x: Tensor) -> Tensor:
        """Extract sliding window context for each block.

        :param x: *Shape:* :math:`(B,T,...)`.
        :returns: *Shape:* :math:`(B,U,C,...)` where C=context_size.
        """
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)

        x_unfolded = x.unfold(
            dimension=1, size=self.context_size, step=self.chunk_size
        )

        # For inputs with extra dims [B, T, N, H], unfold gives
        # [B, U, N, H, C] - move C to position 2 → [B, U, C, N, H]
        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = torch.movedim(
                x_unfolded, source=-1, destination=2
            )

        return x_unfolded.contiguous()

    def _create_local_causal_valid_mask(self) -> Tensor:
        """Create combined local + causal attention mask.

        :returns: Boolean mask. *Shape:* :math:`(W,C)` where
            True = allowed position.
        """
        lower_causal = torch.tril(
            torch.ones(
                (self.context_size, self.chunk_size), dtype=torch.bool
            ),
            diagonal=0,
        ).T
        upper_causal = torch.tril(
            torch.ones(
                (self.chunk_size, self.context_size), dtype=torch.bool
            ),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        return (
            torch.ones(
                (self.chunk_size, self.context_size), dtype=torch.bool
            )
            * lower_causal
            * upper_causal
        )

    def reset_non_persistent_buffers(self) -> None:
        """Re-initialize non-persistent buffers after checkpoint load."""
        self.q_scale.fill_(
            self.head_dim**-0.5 / F.softplus(torch.tensor(0.0)).item()
        )
        self.softcap.fill_(self.logit_cap)
        self.local_causal_valid_mask.copy_(
            self._create_local_causal_valid_mask()
        )
        num_timescales = self.model_dim // 2
        log_inc = math.log(1.0e4) / max(num_timescales - 1, 1)
        inv_ts = torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_inc
        )
        self.inv_timescales.copy_(inv_ts.unsqueeze(0).unsqueeze(0))

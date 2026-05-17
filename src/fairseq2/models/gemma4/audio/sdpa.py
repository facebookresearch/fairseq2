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
class Gemma4ConformerSDPA(Module):
    """Chunked local attention for Gemma4 audio conformer.

    Uses base-2 log scaling with per-dimension learned scale on Q:

      Q *= (head_dim^(-0.5) / ln(2)) * softplus(per_dim_scale)
      K *= ln(1+e) / ln(2)

    Position embeddings are received externally (computed once by the tower
    and shared across layers), unlike Gemma3n where each SDPA computes its own.
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
        chunk_size: int,
        left_context: int,
        right_context: int,
        logit_cap: float,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
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
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.logit_cap = logit_cap

        # Position projection (relative_k_proj in HF)
        self.pos_proj = Linear(
            model_dim,
            num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Per-dimension scale (zeros init, applied as softplus to Q)
        self.per_dim_scale = Parameter(
            torch.zeros(self.head_dim, device=device, dtype=dtype)
        )

        # Gemma4 uses base-2 log scaling: q_scale = head_dim^(-0.5) / log(2)
        q_scale = self.head_dim**-0.5 / math.log(2.0)
        self.register_buffer(
            "q_scale", torch.tensor(q_scale, dtype=torch.float32), persistent=False
        )

        # K gets a constant scale: ln(1+e) / ln(2)
        k_scale = math.log(1 + math.e) / math.log(2.0)
        self.register_buffer(
            "k_scale", torch.tensor(k_scale, dtype=torch.float32), persistent=False
        )

        # Sinusoidal inverse timescales
        num_timescales = model_dim // 2
        log_timescale_increment = math.log(1.0e4) / max(num_timescales - 1, 1)
        inv_timescales = torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer(
            "inv_timescales",
            inv_timescales.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        # Precomputed local causal mask [chunk_size, context_size]
        local_mask = self._create_local_causal_valid_mask()
        self.register_buffer("local_causal_valid_mask", local_mask, persistent=False)

        # Softcap buffer
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
                "Gemma4 conformer SDPA does not support packed batches."
            )

        batch_size, q_time, num_heads, head_dim = q.shape
        input_dtype = q.dtype

        # Cast to float32 for attention computation (matching HF behavior).
        q = q.float()
        k = k.float()
        v = v.float()

        # Gemma4 scaling: Q gets q_scale * softplus(per_dim_scale),
        # K gets constant k_scale = ln(1+e)/ln(2).
        q = q * self.q_scale * F.softplus(self.per_dim_scale)  # type: ignore[operator]
        k = k * self.k_scale  # type: ignore[operator]

        # Convert to blocks
        query_blocks = self._convert_to_block(q)
        key_blocks = self._extract_block_context(k)
        value_blocks = self._extract_block_context(v)
        num_query_blocks = query_blocks.shape[1]

        # Compute logits with relative position embeddings
        logits = self._compute_relative_logits(
            query_blocks, key_blocks, num_query_blocks, input_dtype
        )

        # Softcap on logits
        softcap_val = self.softcap.to(logits.device)  # type: ignore[operator]
        logits = torch.tanh(logits / softcap_val) * softcap_val  # type: ignore[operator]

        # Local causal mask: ALWAYS applied.
        # HF's Gemma4AudioModel.forward() always creates a
        # bidirectional + sliding_window attention mask via
        # create_bidirectional_mask, even when no external mask is
        # provided.  Our local_causal_valid_mask encodes the same
        # constraints for the blocked attention layout.
        causal_condition = (
            self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # type: ignore[operator]
        )  # [1, 1, 1, W, C]

        # Always create validity masks for block-alignment padding.
        # When the sequence length is not a multiple of chunk_size,
        # _convert_to_block zero-pads to the next multiple.  HF's mask
        # marks these padded positions as invalid for BOTH queries and
        # keys; we replicate this exactly.
        padded_len = num_query_blocks * self.chunk_size
        seq_valid = (
            torch.arange(padded_len, device=logits.device) < q_time
        ).unsqueeze(0).expand(batch_size, -1)  # [B, padded_len]

        if mask is not None:
            # Merge explicit padding mask with sequence-length mask.
            # mask shape may be [B, q_time]; pad to padded_len with True (=masked).
            if mask.shape[1] < padded_len:
                mask_padded = F.pad(mask, (0, padded_len - mask.shape[1]), value=True)
            else:
                mask_padded = mask[:, :padded_len]
            seq_valid = seq_valid & (~mask_padded)

        # Key validity: [B, num_blocks, C] → [B, 1, num_blocks, 1, C]
        key_valid = self._extract_block_context(seq_valid)
        key_validity = key_valid.unsqueeze(1).unsqueeze(-2)

        # Query validity: [B, padded_len] → [B, num_blocks, W] → [B, 1, num_blocks, W, 1]
        query_valid = self._convert_to_block(
            seq_valid.unsqueeze(-1)  # [B, padded_len, 1] to match 3D for _convert_to_block
        ).squeeze(-1)  # [B, num_blocks, W]
        query_validity = query_valid.unsqueeze(1).unsqueeze(-1)

        final_condition = causal_condition.to(key_validity.device) & key_validity & query_validity

        logits = torch.where(
            final_condition,
            logits,
            torch.tensor(-1.0e9, dtype=logits.dtype, device=logits.device),
        )

        # Softmax in float32 + weighted sum
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

        # Context vectors via batched matmul
        b, n, u, w, c = probs.shape
        h = value_blocks.shape[-1]
        prob_bun = probs.permute(0, 2, 1, 3, 4).reshape(-1, w, c)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c, h)
        result = torch.bmm(prob_bun, v_bun)
        context = (
            result.reshape(b, u, n, w, h).permute(0, 1, 3, 2, 4).reshape(b, u * w, n, h)
        )
        context = context[:, :q_time]

        return context.to(input_dtype), None

    if TYPE_CHECKING:
        __call__ = forward

    def _compute_relative_logits(
        self,
        query_blocks: Tensor,
        key_blocks: Tensor,
        num_query_blocks: int,
        input_dtype: torch.dtype,
    ) -> Tensor:
        """Compute attention logits with sinusoidal relative positions."""
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

        sin_emb = self._get_timing_signal(pos_indices, dtype=input_dtype)
        projected = self.pos_proj(sin_emb)
        # Cast to float32 to match Q/K dtype for attention computation
        sin_emb_heads = projected.reshape(1, f_span, num_heads, head_dim).squeeze(0).float()

        # Content-content: Q @ K^T
        queries_p = query_blocks.permute(0, 3, 1, 2, 4)
        keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
        term_ac = torch.matmul(queries_p, keys_p_t)

        # Content-position: Q @ pos_emb^T
        s_permuted = sin_emb_heads.permute(1, 2, 0)
        q_reshaped = queries_p.reshape(
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

    def _get_timing_signal(self, position: Tensor, dtype: torch.dtype) -> Tensor:
        """Compute sinusoidal timing signal.

        Matches HF's Gemma4AudioRelPositionalEncoding which computes in the
        model's native dtype (bf16), NOT in float32. This is important for
        parity: bf16 sin/cos gives different results than float32→cast.
        """
        # HF: position_ids[..., None] * inv_timescales  (both in model dtype)
        position_col = position.to(dtype).unsqueeze(-1)
        inv_ts = self.inv_timescales.to(device=position.device, dtype=dtype)  # type: ignore[operator]
        scaled_time = position_col * inv_ts  # type: ignore[operator]
        timing_signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1
        )
        return timing_signal

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
        """Apply relative shift to align position embeddings with keys."""
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

    def _pad_dim1(self, x: Tensor, pad_left: int, pad_right: int) -> Tensor:
        """Zero-pad tensor along dimension 1."""
        batch = x.shape[0]
        tail_shape = x.shape[2:]
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        return torch.cat([left, x, right], dim=1)

    def _convert_to_block(self, x: Tensor) -> Tensor:
        """Split sequence into non-overlapping blocks."""
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        padding_len = num_blocks * self.chunk_size - t
        if padding_len > 0:
            x = self._pad_dim1(x, 0, padding_len)

        new_shape = (b, num_blocks, self.chunk_size) + shape[2:]
        return x.reshape(new_shape).contiguous()

    def _extract_block_context(self, x: Tensor) -> Tensor:
        """Extract sliding window context for each block."""
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)

        x_unfolded = x.unfold(dimension=1, size=self.context_size, step=self.chunk_size)

        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def _create_local_causal_valid_mask(self) -> Tensor:
        """Create combined local + causal attention mask.

        Matches HF's sliding_window_mask_function which uses strict
        ``dist < left_window_size`` (NOT ``<=``).  With chunk_size=12
        and max_past_horizon=12, each query attends to 12 key positions
        (itself + 11 prior), NOT 13.

        Using ``diagonal=-1`` in the lower triangle gives ``j > i``
        (strict), which corresponds to ``dist < max_past_horizon``.
        """
        lower_causal = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=-1,
        ).T
        upper_causal = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        return (
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool)
            * lower_causal
            * upper_causal
        )

    def reset_non_persistent_buffers(self) -> None:
        """Re-initialize non-persistent buffers after checkpoint load."""
        self.q_scale.fill_(self.head_dim**-0.5 / math.log(2.0))  # type: ignore[operator]
        self.k_scale.fill_(math.log(1 + math.e) / math.log(2.0))  # type: ignore[operator]
        self.softcap.fill_(self.logit_cap)  # type: ignore[operator]
        self.local_causal_valid_mask.copy_(self._create_local_causal_valid_mask())  # type: ignore[operator]
        num_timescales = self.model_dim // 2
        log_inc = math.log(1.0e4) / max(num_timescales - 1, 1)
        inv_ts = torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_inc)
        self.inv_timescales.copy_(inv_ts.unsqueeze(0).unsqueeze(0))  # type: ignore[operator]

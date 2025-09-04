# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast, final

import torch
from torch import Tensor
from typing_extensions import override

try:
    import flash_attn_3_cuda  # type: ignore[import-not-found]
except ImportError:
    _has_flash_attn_3 = False
else:
    _has_flash_attn_3 = True

from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout


@final
class Flash3SDPA(SDPA):
    """Computes scaled dot-product attention using FlashAttention3."""

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if not _has_flash_attn_3:
            raise OperationalError(
                "FlashAttention3 is not found. Follow instructions at https://github.com/Dao-AILab/flash-attention."
            )

        if q_layout.padded or k_layout.padded:
            raise NotSupportedError(f"`{Flash3SDPA}` does not support padded batches.")

        if isinstance(self.bias, IdentityBias):
            causal = False

            lhs_window_size = -1
        elif isinstance(self.bias, CausalAttentionBias):
            causal = True

            attn_window_len = self.bias.attn_window_len
            if attn_window_len is not None:
                lhs_window_size = attn_window_len
            else:
                lhs_window_size = -1
        else:
            raise NotSupportedError(f"`{Flash3SDPA}` does not support `{self.bias}`.")

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        if dropout_p != 0.0:
            raise NotSupportedError(f"`{Flash3SDPA}` does not support dropout.")

        if q_layout.packed ^ k_layout.packed:
            raise ValueError("`q_layout` and `k_layout` must be both packed.")

        if q_layout.packed:
            attns = flash_attn_3_varlen(
                q,
                k,
                v,
                q_layout.seq_begin_indices_pt,
                k_layout.seq_begin_indices_pt,
                q_layout.max_seq_len,
                k_layout.max_seq_len,
                causal=causal,
                lhs_window_size=lhs_window_size,
            )
        else:
            attns = flash_attn_3(
                q, k, v, causal=causal, lhs_window_size=lhs_window_size
            )

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"


def flash_attn_3(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    lhs_window_size: int = -1,
    rhs_window_size: int = -1,
) -> Tensor:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    attns, _ = _flash_attn_3_op(
        q, k, v, softmax_scale, causal, lhs_window_size, rhs_window_size
    )

    return cast(Tensor, attns)


@torch.library.custom_op(
    "fairseq2::_flash_attn_3", mutates_args=(), device_types="cuda"
)
def _flash_attn_3_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float | None,
    causal: bool,
    lhs_window_size: int,
    rhs_window_size: int,
) -> tuple[Tensor, Tensor]:
    q = _contiguous(q)
    k = _contiguous(k)
    v = _contiguous(v)

    # fmt: off
    out, softmax_lse, *_ = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        None, # k_new
        None, # v_new
        None, # qv
        None, # out
        None, # cu_seqlens_q
        None, # cu_seqlens_k
        None, # cu_seqlens_k_new
        None, # seqused_q
        None, # seqused_k
        None, # max_seqlen_q
        None, # max_seqlen_k
        None, # page_table
        None, # kv_batch_idx
        None, # leftpad_k
        None, # rotary_cos
        None, # rotary_sin
        None, # seqlens_rotary
        None, # q_descale
        None, # k_descale
        None, # v_descale
        softmax_scale,
        causal,
        lhs_window_size,
        rhs_window_size,
        0,    # attention_chunk
        0.0,  # softcap
        True, # rotary_interleaved
        None, # scheduler_metadata
        1,    # num_splits
        None, # pack_gqa
        0,    # sm_margin
    )
    # fmt: on

    return out, softmax_lse


@_flash_attn_3_op.register_fake
def _flash_attn_3_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float | None,
    causal: bool,
    lhs_window_size: int,
    rhs_window_size: int,
) -> tuple[Tensor, Tensor]:
    out = torch.empty_like(q)

    batch_size, seqlen_q, num_heads, head_size = q.shape

    softmax_shp = (batch_size, num_heads, seqlen_q)

    softmax_lse = torch.empty(
        softmax_shp, dtype=torch.float32, device=q.device, layout=q.layout
    )

    return out, softmax_lse


def _flash_attn_3_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    q, k, v, softmax_scale, causal, lhs_window_size, rhs_window_size = inputs

    out, softmax_lse = output

    ctx.save_for_backward(q, k, v, out, softmax_lse)

    ctx.mark_non_differentiable(softmax_lse)

    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.lhs_window_size = lhs_window_size
    ctx.rhs_window_size = rhs_window_size


def _flash_attn_3_bwd(ctx: Any, dout: Tensor, *_: Any) -> Any:
    q, k, v, out, softmax_lse = ctx.saved_tensors

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(q)

    dout = _contiguous(dout)

    _flash_attn_3_bwd_op(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        ctx.softmax_scale,
        ctx.causal,
        ctx.lhs_window_size,
        ctx.rhs_window_size,
    )

    dq = dq[..., : q.shape[-1]]
    dk = dk[..., : k.shape[-1]]
    dv = dv[..., : v.shape[-1]]

    return dq, dk, dv, None, None, None, None


_flash_attn_3_op.register_autograd(_flash_attn_3_bwd, setup_context=_flash_attn_3_ctx)


def flash_attn_3_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    lhs_window_size: int = -1,
    rhs_window_size: int = -1,
) -> Tensor:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    attns, _ = _flash_attn_3_varlen_op(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        lhs_window_size,
        rhs_window_size,
    )

    return cast(Tensor, attns)


@torch.library.custom_op(
    "fairseq2::_flash_attn_3_varlen", mutates_args=(), device_types="cuda"
)
def _flash_attn_3_varlen_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    lhs_window_size: int,
    rhs_window_size: int,
) -> tuple[Tensor, Tensor]:
    q = _contiguous(q)
    k = _contiguous(k)
    v = _contiguous(v)

    cu_seqlens_q = _contiguous(cu_seqlens_q)
    cu_seqlens_k = _contiguous(cu_seqlens_k)

    # fmt: off
    out, softmax_lse, *_ = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        None, # k_new
        None, # v_new
        None, # qv
        None, # out
        cu_seqlens_q,
        cu_seqlens_k,
        None, # cu_seqlens_k_new
        None, # seqused_q
        None, # seqused_k
        max_seqlen_q,
        max_seqlen_k,
        None, # page_table
        None, # kv_batch_idx
        None, # leftpad_k
        None, # rotary_cos
        None, # rotary_sin
        None, # seqlens_rotary
        None, # q_descale
        None, # k_descale
        None, # v_descale
        softmax_scale,
        causal,
        lhs_window_size,
        rhs_window_size,
        0,    # attention_chunk
        0.0,  # softcap
        True, # rotary_interleaved
        None, # scheduler_metadata
        1,    # num_splits
        None, # pack_gqa
        0,    # sm_margin
    )
    # fmt: on

    return out, softmax_lse


@_flash_attn_3_varlen_op.register_fake
def _flash_attn_3_varlen_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    lhs_window_size: int,
    rhs_window_size: int,
) -> tuple[Tensor, Tensor]:
    out = torch.empty_like(q)

    seqlen_q, num_heads, head_size = q.shape

    softmax_lse = torch.empty(
        (num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout
    )

    return out, softmax_lse


def _flash_attn_3_varlen_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        lhs_window_size,
        rhs_window_size,
    ) = inputs

    out, softmax_lse = output

    ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k, out, softmax_lse)

    ctx.mark_non_differentiable(softmax_lse)

    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.lhs_window_size = lhs_window_size
    ctx.rhs_window_size = rhs_window_size


def _flash_attn_3_varlen_bwd(ctx: Any, dout: Tensor, *_: Any) -> Any:
    q, k, v, cu_seqlens_q, cu_seqlens_k, out, softmax_lse = ctx.saved_tensors

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(q)

    dout = _contiguous(dout)

    _flash_attn_3_bwd_op(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.softmax_scale,
        ctx.causal,
        ctx.lhs_window_size,
        ctx.rhs_window_size,
    )

    dq = dq[..., : q.shape[-1]]
    dk = dk[..., : k.shape[-1]]
    dv = dv[..., : v.shape[-1]]

    return dq, dk, dv, None, None, None, None, None, None, None, None


_flash_attn_3_varlen_op.register_autograd(
    _flash_attn_3_varlen_bwd, setup_context=_flash_attn_3_varlen_ctx
)


@torch.library.custom_op(
    "fairseq2::_flash_attn_3_bwd", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
def _flash_attn_3_bwd_op(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    cu_seqlens_q: Tensor | None,
    cu_seqlens_k: Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    softmax_scale: float,
    causal: bool,
    lhs_window_size: int,
    rhs_window_size: int,
) -> None:
    # fmt: off
    flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        None, # seqused_q
        None, # seqused_k
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        lhs_window_size,
        rhs_window_size,
        0.0,   # softcap
        False, # deterministic,
        0,     # sm_margin,
    )
    # fmt: on


def _contiguous(x: Tensor) -> Tensor:
    return x.contiguous() if x.stride(-1) != 1 else x

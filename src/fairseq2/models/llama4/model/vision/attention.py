# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from fairseq2.nn import Linear, Projection


def rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    def _norm(y: torch.Tensor) -> torch.Tensor:
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


def apply_scaling(
    freqs: torch.Tensor, scale_factor: float, high_freq_factor: float
) -> torch.Tensor:
    low_freq_factor = 1
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool,
    scale_factor: float,
    high_freq_factor: float,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    # TODO: this module needs to be phased out in favor of StandardMultiheadAttention
    wq: Projection
    wk: Projection
    wv: Projection
    wo: Projection

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        use_qk_norm: bool,
        use_rope: bool,
        add_bias: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        # For attention temperature tuning
        self.attn_temperature_tuning = False
        self.floor_scale = 8192.0
        self.attn_scale = 0.1

        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        # Will get divided when sharding
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        # In the ref implementation, these are hard-coded
        # They only concern the vision encoder though, not the L4 decoder
        self.max_batch_size = 32
        self.max_seq_len = 2048
        norm_eps = 1e-5

        self.wq = Linear(
            dim,
            n_heads * self.head_dim,
            bias=add_bias,
            init_fn=lambda x: None,
        )
        self.wk = Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            init_fn=lambda x: None,
        )
        self.wv = Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            init_fn=lambda x: None,
        )
        self.wo = Linear(
            n_heads * self.head_dim,
            dim,
            bias=add_bias,
            init_fn=lambda x: None,
        )

        self.init_kv_cache()

        self.norm_eps = norm_eps
        self._register_load_state_dict_pre_hook(self.load_hook)

    def init_kv_cache(self) -> None:
        self.cache_k = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "wqkv.weight" in state_dict:
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            d, r = divmod(wqkv.shape[0], self.n_heads + 2 * self.n_kv_heads)
            if r != 0:
                raise ValueError(
                    f"shape={tuple(wqkv.shape)} is not divisible by "
                    f"n_heads ({self.n_heads}) + 2 * n_kv_heads ({self.n_kv_heads})"
                )
            wq, wk, wv = wqkv.split(
                [d * self.n_heads, d * self.n_kv_heads, d * self.n_kv_heads], dim=0
            )
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.use_qk_norm:
            xq = rmsnorm(xq, self.norm_eps)
            xk = rmsnorm(xk, self.norm_eps)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        if self.attn_temperature_tuning and not self.use_rope:
            seq_positions = torch.arange(
                start_pos, start_pos + seqlen, device=xq.device, dtype=torch.float32
            )
            attn_scales = (
                torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) + 1.0)
                * self.attn_scale
                + 1.0
            )

            # reshape for broadcasting [seqlen] -> [1, seqlen, 1, 1]
            attn_scales = attn_scales.view(1, seqlen, 1, 1)
            xq = xq * attn_scales

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        xk = self.cache_k[:bsz, : start_pos + seqlen]
        xv = self.cache_v[:bsz, : start_pos + seqlen]

        xq, xk, xv = [t.transpose(1, 2) for t in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output: torch.Tensor = self.wo(attn_output)
        return output

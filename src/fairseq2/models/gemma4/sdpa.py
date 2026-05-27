# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma4-specific scaled dot-product attention.

Gemma4 uses QK-norm and sets ``scale=1.0`` to disable the default
``1/sqrt(head_dim)`` scaling.  The shared :class:`TorchSDPA` implements custom
scaling by pre-multiplying Q: ``Q *= scale * sqrt(head_dim)``.  In bfloat16 this
amplification (e.g. ``Q * sqrt(512) ≈ Q * 22.6``) introduces precision loss
that is benign for dense models but gets amplified by MoE routing, causing
severe divergence in models like 26B-A4B.

This module passes ``scale`` directly to PyTorch's
:func:`scaled_dot_product_attention` (available since PyTorch 2.1) which applies
it in the fused kernel without Q magnification, preserving bfloat16 precision.
"""

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    maybe_get_attention_bias_tensor,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout


@final
class Gemma4SDPA(SDPA):
    """Scaled dot-product attention for Gemma4 models.

    Unlike :class:`TorchSDPA`, this passes ``scale`` directly to PyTorch's
    ``scaled_dot_product_attention`` kernel instead of pre-scaling Q.  This
    avoids bfloat16 precision loss from Q magnification, which is critical for
    MoE models where small attention differences cascade through expert routing.
    """

    bias: AttentionBias
    dropout_p: float
    scale: float | None

    def __init__(
        self, bias: AttentionBias, *, dropout_p: float = 0.0, scale: float | None = None
    ) -> None:
        """
        :param bias:
            The attention bias.
        :param dropout_p:
            The dropout probability for attention weights.
        :param scale:
            The scaling factor to apply to attention logits.  If ``None``, uses
            PyTorch's default ``1/sqrt(head_dim)`` scaling.  Set to ``1.0`` to
            disable scaling (e.g. when using QK normalization).

            Unlike :class:`TorchSDPA`, this value is passed directly to
            ``scaled_dot_product_attention(scale=...)`` so no Q pre-scaling is
            needed.
        """
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p
        self.scale = scale

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
        if needs_weights:
            raise NotSupportedError(f"`{Gemma4SDPA}` does not support `needs_weights`.")

        is_causal = False

        # ([[N], H], S, S_kv)
        if isinstance(self.bias, CausalAttentionBias):
            if self.bias.attn_window_len is None:
                full_q = not q_layout.packed and not q_layout.padded
                full_k = not k_layout.packed and not k_layout.padded

                if full_q and full_k:
                    q_len = q.size(1)
                    k_len = k.size(1)

                    is_causal = q_len == k_len

        if is_causal:
            bias = None
        else:
            # ([[N], H], S, S_kv)
            bias = maybe_get_attention_bias_tensor(
                self.bias, q, q_layout, k_layout, bias_cache
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        # (N, S, H, K) -> (N, H, S, K)
        q = q.transpose(-2, -3)

        # (N, S_kv, H, K) -> (N, H, S_kv, K)
        k = k.transpose(-2, -3)

        # (N, S_kv, H, V) -> (N, H, S_kv, V)
        v = v.transpose(-2, -3)

        # (N, H, S, V)
        # Pass scale directly to the kernel — no Q pre-multiplication needed.
        attns = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.scale,
        )

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, None

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"bias={self.bias}, dropout_p={self.dropout_p:G}"
        if self.scale is not None:
            s += f", scale={self.scale:G}"
        return s

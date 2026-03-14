# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, SiLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.conformer import ConformerConvolution
from fairseq2.models.gemma3n.audio.sdpa import Gemma3nConformerSDPA
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig
from fairseq2.models.transformer import (
    AttentionBiasCache,
    StandardFeedForwardNetwork,
    TransformerEncoderLayer,
)
from fairseq2.nn import BatchLayout, RMSNorm
from fairseq2.nn.projection import Linear


@final
class Gemma3nConformerAttention(Module):
    """Self-attention for Gemma3n conformer with chunked local attention.

    Owns Q/K/V projections and a :class:`Gemma3nConformerSDPA` instance.
    Passes mask directly to SDPA (no side channel).
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear
    sdpa: Gemma3nConformerSDPA
    num_heads: int
    head_dim: int

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: Gemma3nConformerSDPA,
        *,
        bias: bool = False,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = Linear(
            model_dim, model_dim, bias=bias, device=device, dtype=dtype,
        )
        self.k_proj = Linear(
            model_dim, model_dim, bias=bias, device=device, dtype=dtype,
        )
        self.v_proj = Linear(
            model_dim, model_dim, bias=bias, device=device, dtype=dtype,
        )
        self.output_proj = Linear(
            model_dim, model_dim, bias=bias, device=device, dtype=dtype,
        )
        self.sdpa = sdpa

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        bias_cache: AttentionBiasCache,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs: Input features. *Shape:* :math:`(N,T,M)`.
        :param seqs_layout: Layout information.
        :param bias_cache: Attention bias cache (unused, kept for API compat).
        :param mask: Where True=masked (invalid). *Shape:* :math:`(N,T)`.
        :returns: Attention output. *Shape:* :math:`(N,T,M)`.
        """
        q = self.q_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))

        attns, _ = self.sdpa(
            q, seqs_layout, k, seqs_layout, v, mask=mask,
        )

        return self.output_proj(attns.flatten(-2, -1))

    if TYPE_CHECKING:
        __call__ = forward


@final
class Gemma3nConformerBlock(TransformerEncoderLayer):
    """Gemma3n conformer block matching HuggingFace architecture.

    Forward flow (per sub-block):
      FFN1:  clamp -> pre_norm -> ffn -> clamp -> post_norm -> *0.5 -> residual
      Attn:  clamp -> pre_norm -> self_attn -> clamp -> post_norm -> residual
      Conv:  (mask input) -> pre_norm -> conv -> residual
      FFN2:  clamp -> pre_norm -> ffn -> clamp -> post_norm -> *0.5 -> residual
      Block: clamp -> layer_norm
    """

    ffn1_layer_norm: RMSNorm
    ffn1: StandardFeedForwardNetwork
    ffn1_post_layer_norm: RMSNorm
    self_attn_layer_norm: RMSNorm
    self_attn: Gemma3nConformerAttention
    self_attn_post_norm: RMSNorm
    conv_layer_norm: RMSNorm
    conv: ConformerConvolution
    ffn2_layer_norm: RMSNorm
    ffn2: StandardFeedForwardNetwork
    ffn2_post_layer_norm: RMSNorm
    layer_norm: RMSNorm
    gradient_clipping: float
    residual_weight: float

    def __init__(
        self,
        *,
        ffn1_layer_norm: RMSNorm,
        ffn1: StandardFeedForwardNetwork,
        ffn1_post_layer_norm: RMSNorm,
        self_attn_layer_norm: RMSNorm,
        self_attn: Gemma3nConformerAttention,
        self_attn_post_norm: RMSNorm,
        conv_layer_norm: RMSNorm,
        conv: ConformerConvolution,
        ffn2_layer_norm: RMSNorm,
        ffn2: StandardFeedForwardNetwork,
        ffn2_post_layer_norm: RMSNorm,
        layer_norm: RMSNorm,
        gradient_clipping: float = 1e10,
        residual_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.ffn1_layer_norm = ffn1_layer_norm
        self.ffn1 = ffn1
        self.ffn1_post_layer_norm = ffn1_post_layer_norm

        self.self_attn_layer_norm = self_attn_layer_norm
        self.self_attn = self_attn
        self.self_attn_post_norm = self_attn_post_norm

        self.conv_layer_norm = conv_layer_norm
        self.conv = conv

        self.ffn2_layer_norm = ffn2_layer_norm
        self.ffn2 = ffn2
        self.ffn2_post_layer_norm = ffn2_post_layer_norm

        self.layer_norm = layer_norm
        self.gradient_clipping = gradient_clipping
        self.residual_weight = residual_weight

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        seqs = self._forward_ffn1(seqs)

        seqs = self._forward_self_attn(seqs, seqs_layout, attn_bias_cache, mask)

        seqs = self._forward_conv(seqs, seqs_layout, mask)

        seqs = self._forward_ffn2(seqs)

        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)

        return self.layer_norm(seqs)

    def _forward_ffn1(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)
        seqs = self.ffn1_layer_norm(seqs)
        seqs = self.ffn1(seqs)
        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)
        seqs = self.ffn1_post_layer_norm(seqs)

        return residual + seqs * self.residual_weight

    def _forward_self_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        mask: Tensor | None,
    ) -> Tensor:
        residual = seqs

        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)
        seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(seqs, seqs_layout, attn_bias_cache, mask=mask)

        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)

        return residual + self.self_attn_post_norm(seqs)

    def _forward_conv(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        mask: Tensor | None,
    ) -> Tensor:
        # Zero out masked positions before conv (matches HF behavior)
        if mask is not None:
            validity = ~mask
            seqs = seqs * validity.unsqueeze(-1).to(seqs.dtype)

        residual = seqs

        seqs = self.conv_layer_norm(seqs)
        seqs = self.conv(seqs, seqs_layout)

        return seqs + residual

    def _forward_ffn2(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)
        seqs = self.ffn2_layer_norm(seqs)
        seqs = self.ffn2(seqs)
        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)
        seqs = self.ffn2_post_layer_norm(seqs)

        return residual + seqs * self.residual_weight


@final
class Gemma3nConformerEncoder(Module):
    """Gemma3n audio encoder using conformer architecture.

    Stacks conformer blocks with chunked local attention, per-dimension
    scaling, and macaron-style FFN. Applies reduction factor downsampling
    after conformer processing.
    """

    layers: ModuleList
    reduction_factor: int

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config: Audio tower configuration.
        """
        super().__init__()

        self.reduction_factor = config.conf_reduction_factor

        layers = []
        for _ in range(config.conf_num_hidden_layers):
            layer = _build_conformer_block(config, device=device, dtype=dtype)
            layers.append(layer)

        self.layers = ModuleList(layers)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs: Audio features. *Shape:* :math:`(N,T,H)`.
        :param seqs_layout: Layout information for the sequences.
        :param mask: Where True=masked (invalid). *Shape:* :math:`(N,T)`.
        :returns: Encoded features. *Shape:* :math:`(N,T/R,H)`.
        """
        bias_cache = AttentionBiasCache()

        for layer in self.layers:
            seqs = layer(seqs, seqs_layout, bias_cache, mask=mask)

        if self.reduction_factor > 1:
            seqs = seqs[:, :: self.reduction_factor]
            if mask is not None:
                mask = mask[:, :: self.reduction_factor]

        if mask is not None:
            seqs = seqs.masked_fill(mask.unsqueeze(-1), 0.0)

        return seqs


def _build_conformer_block(
    config: Gemma3nAudioConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Gemma3nConformerBlock:
    """Build a single Gemma3n conformer block."""
    inner_dim = config.hidden_size * 4

    ffn1 = StandardFeedForwardNetwork(
        model_dim=config.hidden_size,
        inner_dim=inner_dim,
        bias=False,
        inner_activation=SiLU(),
        device=device,
        dtype=dtype,
    )

    ffn1_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn1_post_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    sdpa = Gemma3nConformerSDPA(
        model_dim=config.hidden_size,
        num_heads=config.conf_num_attention_heads,
        max_left_rel_pos=config.conf_attention_context_left,
        max_right_rel_pos=config.conf_attention_context_right,
        chunk_size=config.conf_attention_chunk_size,
        left_context=config.conf_attention_context_left,
        right_context=config.conf_attention_context_right,
        logit_cap=config.conf_attention_logit_cap,
        device=device,
        dtype=dtype,
    )

    self_attn = Gemma3nConformerAttention(
        model_dim=config.hidden_size,
        num_heads=config.conf_num_attention_heads,
        sdpa=sdpa,
        bias=False,
        device=device,
        dtype=dtype,
    )

    self_attn_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    self_attn_post_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    conv = ConformerConvolution(
        model_dim=config.hidden_size,
        depthwise_kernel_size=config.conf_conv_kernel_size,
        causal_depthwise_conv=True,
        norm_type="layer_norm",
        device=device,
        dtype=dtype,
    )
    # Replace StandardLayerNorm (bias=True) with RMSNorm (no bias)
    # to match HF's Gemma3nRMSNorm used in conv_norm
    conv.layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    conv_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn2 = StandardFeedForwardNetwork(
        model_dim=config.hidden_size,
        inner_dim=inner_dim,
        bias=False,
        inner_activation=SiLU(),
        device=device,
        dtype=dtype,
    )

    ffn2_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn2_post_layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    layer_norm = RMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    return Gemma3nConformerBlock(
        ffn1_layer_norm=ffn1_layer_norm,
        ffn1=ffn1,
        ffn1_post_layer_norm=ffn1_post_layer_norm,
        self_attn_layer_norm=self_attn_layer_norm,
        self_attn=self_attn,
        self_attn_post_norm=self_attn_post_norm,
        conv_layer_norm=conv_layer_norm,
        conv=conv,
        ffn2_layer_norm=ffn2_layer_norm,
        ffn2=ffn2,
        ffn2_post_layer_norm=ffn2_post_layer_norm,
        layer_norm=layer_norm,
        gradient_clipping=config.gradient_clipping,
        residual_weight=config.conf_residual_weight,
    )

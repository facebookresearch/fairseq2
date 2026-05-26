# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import GLU, Conv1d, Module, ModuleList, SiLU
from torch.nn.functional import pad
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma4.audio.clipped_linear import Gemma4ClippedLinear
from fairseq2.models.gemma4.audio.config import Gemma4AudioConfig
from fairseq2.models.gemma4.audio.norm import Gemma4AudioRMSNorm
from fairseq2.models.gemma4.audio.sdpa import Gemma4ConformerSDPA
from fairseq2.models.transformer import (
    AttentionBiasCache,
    TransformerEncoderLayer,
)
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.mask import apply_mask


# ---------------------------------------------------------------------------
# Gemma4AudioFFN — replaces StandardFeedForwardNetwork with ClippedLinear
# ---------------------------------------------------------------------------
@final
class Gemma4AudioFFN(Module):
    """Gemma4 audio feed-forward network using ClippedLinear.

    Matches HF ``Gemma4AudioFeedForward``'s inner linear ops:
      ClippableLinear(d, 4d) -> SiLU -> ClippableLinear(4d, d)

    The pre/post layer norms, gradient clipping, and residual connection
    are handled externally in ``Gemma4ConformerBlock``.
    """

    inner_proj: Gemma4ClippedLinear
    inner_activation: SiLU
    output_proj: Gemma4ClippedLinear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        *,
        bias: bool = False,
        use_clipping: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.inner_proj = Gemma4ClippedLinear(
            model_dim,
            inner_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )
        self.inner_activation = SiLU()
        self.output_proj = Gemma4ClippedLinear(
            inner_dim,
            model_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )

    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.inner_proj(seqs)
        seqs = self.inner_activation(seqs)
        seqs = self.output_proj(seqs)
        return seqs

    if TYPE_CHECKING:
        __call__ = forward


# ---------------------------------------------------------------------------
# Gemma4AudioConvModule — replaces ConformerConvolution with ClippedLinear
# ---------------------------------------------------------------------------
@final
class Gemma4AudioConvModule(Module):
    """Gemma4 audio conformer convolution module.

    Matches HF ``Gemma4AudioLightConv1d`` (excluding pre_layer_norm and residual):
      ClippableLinear(d, 2d) -> GLU -> depthwise Conv1d -> clamp ->
      RMSNorm -> SiLU -> ClippableLinear(d, d)

    Uses ``Gemma4ClippedLinear`` for pointwise ops (operating on last dim)
    and plain ``Conv1d`` for the depthwise convolution.
    """

    pointwise_conv1: Gemma4ClippedLinear
    pointwise_conv1_activation: GLU
    depthwise_conv: Conv1d
    causal_depthwise_conv: bool
    layer_norm: Gemma4AudioRMSNorm
    depthwise_activation: SiLU
    pointwise_conv2: Gemma4ClippedLinear
    gradient_clipping: float

    def __init__(
        self,
        model_dim: int,
        depthwise_kernel_size: int,
        *,
        causal_depthwise_conv: bool = False,
        gradient_clipping: float = 1e10,
        rms_norm_eps: float = 1e-6,
        use_clipping: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.gradient_clipping = gradient_clipping
        self.causal_depthwise_conv = causal_depthwise_conv

        # Pointwise conv 1: Linear(d, 2d) + GLU → d
        self.pointwise_conv1 = Gemma4ClippedLinear(
            model_dim,
            model_dim * 2,
            bias=False,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )
        self.pointwise_conv1_activation = GLU(dim=-1)

        # Depthwise conv: Conv1d with groups=d
        self.depthwise_conv = Conv1d(
            model_dim,
            model_dim,
            depthwise_kernel_size,
            padding="same" if not causal_depthwise_conv else 0,
            groups=model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Layer norm after depthwise conv (RMSNorm to match HF)
        self.layer_norm = Gemma4AudioRMSNorm(
            model_dim,
            bias=False,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.depthwise_activation = SiLU()

        # Pointwise conv 2: Linear(d, d)
        self.pointwise_conv2 = Gemma4ClippedLinear(
            model_dim,
            model_dim,
            bias=False,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """
        :param seqs: *Shape:* :math:`(N,S,M)`.
        :returns: Processed sequences. *Shape:* :math:`(N,S,M)`.
        """
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if seqs_layout.padded:
            padding_mask = seqs_layout.position_indices >= 0
            seqs = apply_mask(seqs, padding_mask)

        # Pointwise conv 1 (Linear on last dim): (N, S, d) -> (N, S, 2d)
        seqs = self.pointwise_conv1(seqs)

        # GLU on last dim: (N, S, 2d) -> (N, S, d)
        seqs = self.pointwise_conv1_activation(seqs)

        # Transpose for depthwise conv: (N, S, d) -> (N, d, S)
        seqs = seqs.transpose(1, 2)

        # Causal padding
        if self.causal_depthwise_conv:
            seqs = pad(seqs, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # Depthwise conv: (N, d, S) -> (N, d, S)
        seqs = self.depthwise_conv(seqs)

        # Transpose back: (N, d, S) -> (N, S, d)
        seqs = seqs.transpose(1, 2)

        # Gradient clipping after depthwise conv (matches HF)
        seqs = torch.clamp(seqs, -self.gradient_clipping, self.gradient_clipping)

        # Layer norm + activation
        seqs = self.layer_norm(seqs)
        seqs = self.depthwise_activation(seqs)

        # Pointwise conv 2 (Linear on last dim): (N, S, d) -> (N, S, d)
        seqs = self.pointwise_conv2(seqs)

        return seqs

    if TYPE_CHECKING:
        __call__ = forward


# ---------------------------------------------------------------------------
# Gemma4ConformerAttention — self-attention with ClippedLinear projections
# ---------------------------------------------------------------------------
@final
class Gemma4ConformerAttention(Module):
    """Self-attention for Gemma4 conformer with chunked local attention.

    All linear projections use ``Gemma4ClippedLinear`` to match HF's
    ``Gemma4ClippableLinear`` wrappers with input/output clamping.
    """

    q_proj: Gemma4ClippedLinear
    k_proj: Gemma4ClippedLinear
    v_proj: Gemma4ClippedLinear
    output_proj: Gemma4ClippedLinear
    sdpa: Gemma4ConformerSDPA
    num_heads: int
    head_dim: int

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: Gemma4ConformerSDPA,
        *,
        bias: bool = False,
        use_clipping: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = Gemma4ClippedLinear(
            model_dim,
            model_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Gemma4ClippedLinear(
            model_dim,
            model_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Gemma4ClippedLinear(
            model_dim,
            model_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Gemma4ClippedLinear(
            model_dim,
            model_dim,
            bias=bias,
            use_clipping=use_clipping,
            device=device,
            dtype=dtype,
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
        q = self.q_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(seqs).unflatten(-1, (self.num_heads, self.head_dim))

        attns, _ = self.sdpa(
            q,
            seqs_layout,
            k,
            seqs_layout,
            v,
            mask=mask,
        )

        return self.output_proj(attns.flatten(-2, -1))  # type: ignore[no-any-return]

    if TYPE_CHECKING:
        __call__ = forward


# ---------------------------------------------------------------------------
# Gemma4ConformerBlock — macaron-style conformer block
# ---------------------------------------------------------------------------
@final
class Gemma4ConformerBlock(TransformerEncoderLayer):
    """Gemma4 conformer block.

    Forward flow::

      FFN1:  clamp -> pre_norm -> ffn -> clamp -> post_norm -> *0.5 -> residual
      Attn:  clamp -> pre_norm -> self_attn -> clamp -> post_norm -> residual
      Conv:  (mask) -> pre_norm -> conv -> residual
      FFN2:  clamp -> pre_norm -> ffn -> clamp -> post_norm -> *0.5 -> residual
      Block: clamp -> layer_norm
    """

    ffn1_layer_norm: Gemma4AudioRMSNorm
    ffn1: Gemma4AudioFFN
    ffn1_post_layer_norm: Gemma4AudioRMSNorm
    self_attn_layer_norm: Gemma4AudioRMSNorm
    self_attn: Gemma4ConformerAttention
    self_attn_post_norm: Gemma4AudioRMSNorm
    conv_layer_norm: Gemma4AudioRMSNorm
    conv: Gemma4AudioConvModule
    ffn2_layer_norm: Gemma4AudioRMSNorm
    ffn2: Gemma4AudioFFN
    ffn2_post_layer_norm: Gemma4AudioRMSNorm
    layer_norm: Gemma4AudioRMSNorm
    gradient_clipping: float
    residual_weight: float

    def __init__(
        self,
        *,
        ffn1_layer_norm: Gemma4AudioRMSNorm,
        ffn1: Gemma4AudioFFN,
        ffn1_post_layer_norm: Gemma4AudioRMSNorm,
        self_attn_layer_norm: Gemma4AudioRMSNorm,
        self_attn: Gemma4ConformerAttention,
        self_attn_post_norm: Gemma4AudioRMSNorm,
        conv_layer_norm: Gemma4AudioRMSNorm,
        conv: Gemma4AudioConvModule,
        ffn2_layer_norm: Gemma4AudioRMSNorm,
        ffn2: Gemma4AudioFFN,
        ffn2_post_layer_norm: Gemma4AudioRMSNorm,
        layer_norm: Gemma4AudioRMSNorm,
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
        return self.layer_norm(seqs)  # type: ignore[no-any-return]

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
        return residual + self.self_attn_post_norm(seqs)  # type: ignore[no-any-return]

    def _forward_conv(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        mask: Tensor | None,
    ) -> Tensor:
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


# ---------------------------------------------------------------------------
# Gemma4ConformerEncoder — stacks conformer blocks
# ---------------------------------------------------------------------------
@final
class Gemma4ConformerEncoder(Module):
    """Gemma4 audio encoder using conformer architecture.

    Unlike Gemma3n, does NOT apply reduction factor downsampling.
    The temporal resolution from subsample (T/4) is preserved.
    """

    layers: ModuleList

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        layers = []
        for _ in range(config.num_hidden_layers):
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
        :returns: Encoded features. *Shape:* :math:`(N,T,H)` — NO reduction.
        """
        bias_cache = AttentionBiasCache()

        for layer in self.layers:
            seqs = layer(seqs, seqs_layout, bias_cache, mask=mask)

        # No reduction factor in Gemma4 (unlike Gemma3n which has 4x)

        if mask is not None:
            seqs = seqs.masked_fill(mask.unsqueeze(-1), 0.0)

        return seqs


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------
def _build_conformer_block(
    config: Gemma4AudioConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Gemma4ConformerBlock:
    """Build a single Gemma4 conformer block."""
    inner_dim = config.hidden_size * 4

    ffn1 = Gemma4AudioFFN(
        model_dim=config.hidden_size,
        inner_dim=inner_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )

    ffn1_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn1_post_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    sdpa = Gemma4ConformerSDPA(
        model_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        chunk_size=config.attention_chunk_size,
        left_context=config.attention_context_left,
        right_context=config.attention_context_right,
        logit_cap=config.attention_logit_cap,
        device=device,
        dtype=dtype,
    )

    self_attn = Gemma4ConformerAttention(
        model_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        sdpa=sdpa,
        bias=False,
        device=device,
        dtype=dtype,
    )

    self_attn_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    self_attn_post_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    conv = Gemma4AudioConvModule(
        model_dim=config.hidden_size,
        depthwise_kernel_size=config.conv_kernel_size,
        causal_depthwise_conv=True,
        gradient_clipping=config.gradient_clipping,
        rms_norm_eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    conv_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn2 = Gemma4AudioFFN(
        model_dim=config.hidden_size,
        inner_dim=inner_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )

    ffn2_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    ffn2_post_layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    layer_norm = Gemma4AudioRMSNorm(
        config.hidden_size,
        bias=False,
        eps=config.rms_norm_eps,
        device=device,
        dtype=dtype,
    )

    return Gemma4ConformerBlock(
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
        residual_weight=config.residual_weight,
    )

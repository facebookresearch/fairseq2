# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma3n.config import Gemma3nConfig, is_global_layer
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerNormOrder,
)
from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork
from fairseq2.models.transformer.sdpa.soft_capped import SoftCappedSDPA
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoderLayer,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import LayerNorm, PositionEncoder, RMSNorm
from fairseq2.nn.position_encoder import DualRotaryEncoder


def create_gemma3n_decoder_layer(
    layer_idx: int,
    config: Gemma3nConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerLMDecoderLayer:
    """Create a Gemma3n decoder layer with local or global attention.

    :param layer_idx: The layer index (0-based).
    :param config: The Gemma3n configuration.
    :returns: A configured decoder layer.
    """
    is_global = is_global_layer(layer_idx, config.num_layers)

    pos_encoder: PositionEncoder | None = None
    if not is_global:
        # Local layers use dual-frequency RoPE
        pos_encoder = DualRotaryEncoder(
            encoding_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            dual_theta=config.rope_theta_global,
            device=device,
        )

    sdpa = SoftCappedSDPA(
        bias=CausalAttentionBias(),
        soft_cap=config.final_logit_soft_cap,
        dropout_p=0.0,
    )

    self_attn = StandardMultiheadAttention(
        model_dim=config.model_dim,
        num_heads=config.num_attn_heads,
        num_key_value_heads=config.num_key_value_heads,
        sdpa=sdpa,
        pos_encoder=pos_encoder,
        bias=False,
        device=device,
        dtype=dtype,
    )

    ffn: FeedForwardNetwork
    if is_global:
        # Global layers use standard GLU FFN
        ffn = GLUFeedForwardNetwork(
            model_dim=config.model_dim,
            inner_dim=config.ffn_inner_dim,
            bias=False,
            gate_activation=nn.GELU(),
            device=device,
            dtype=dtype,
        )
    else:
        # Local layers use AltUp FFN
        ffn = AltUpFeedForwardNetwork(
            model_dim=config.model_dim,
            inner_dim=config.altup_hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

    self_attn_layer_norm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    ffn_layer_norm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )

    return StandardTransformerLMDecoderLayer(
        self_attn=self_attn,
        self_attn_layer_norm=self_attn_layer_norm,
        ffn=ffn,
        ffn_layer_norm=ffn_layer_norm,
        norm_order=TransformerNormOrder.PRE,
        dropout_p=0.0,
        device=device,
        dtype=dtype,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.gemma3n.config import Gemma3nConfig, is_global_layer
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerNormOrder,
)
from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork
from fairseq2.models.transformer.sdpa.soft_capped import SoftCappedSDPA
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    StandardTransformerLMDecoderLayer,
    TransformerLM,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    RMSNorm,
    StandardEmbedding,
    TiedProjection,
)
from fairseq2.nn.position_encoder import DualRotaryEncoder


def create_gemma3n_model(
    config: Gemma3nConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerLM:
    """Create a Gemma3n language model.

    :param config: The Gemma3n configuration.
    :param device: The device on which to initialize the model.
    :param dtype: The data type of the model parameters and buffers.
    :returns: A Gemma3n model.
    """
    gangs = maybe_get_current_gangs()

    return Gemma3nFactory(config, device=device, dtype=dtype, gangs=gangs).create_model()


class Gemma3nFactory:
    """Factory for creating Gemma3n model components."""

    _config: Gemma3nConfig
    _device: Device | None
    _dtype: DataType | None
    _gangs: Gangs | None

    def __init__(
        self,
        config: Gemma3nConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        gangs: Gangs | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._dtype = dtype
        self._gangs = gangs

    def create_model(self) -> TransformerLM:
        """Create the full Gemma3n model."""
        embed = self.create_embedding()
        decoder_frontend = self.create_decoder_frontend(embed)
        decoder = self.create_decoder()
        final_proj = self.create_final_projection(embed)

        return TransformerLM(
            self._config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            self._config.pad_idx,
            self._config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        """Create the token embedding layer."""
        return StandardEmbedding(
            self._config.vocab_size,
            self._config.model_dim,
            self._config.pad_idx,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Create the decoder frontend (embedding layer)."""
        return TransformerEmbeddingFrontend(
            self._config.model_dim,
            embed,
            pos_encoder=None,
            no_scale=True,
            dropout_p=0.0,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder(self) -> TransformerLMDecoder:
        """Create the decoder stack."""
        layers = [
            create_gemma3n_decoder_layer(
                idx, self._config, device=self._device, dtype=self._dtype
            )
            for idx in range(self._config.num_layers)
        ]

        layer_norm = RMSNorm(
            self._config.model_dim,
            bias=False,
            eps=self._config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_final_projection(self, embed: Embedding) -> Projection:
        """Create the final output projection."""
        if not isinstance(embed, StandardEmbedding):
            raise TypeError(
                f"`embed` must be `StandardEmbedding`, got `{type(embed)}` instead."
            )

        return TiedProjection(embed.weight, bias=None)


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

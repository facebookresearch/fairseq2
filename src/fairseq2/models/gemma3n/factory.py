# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.gemma3n.altup import Gemma3nAltUp
from fairseq2.models.gemma3n.config import (
    Gemma3nConfig,
    get_kv_projection_role,
    is_global_layer,
)
from fairseq2.models.gemma3n.decoder import Gemma3nDecoder
from fairseq2.models.gemma3n.decoder_layer import (
    Gemma3nDecoderLayer,
    Gemma3nLAuReL,
)
from fairseq2.models.gemma3n.frontend import Gemma3nFrontend
from fairseq2.models.gemma3n.model import Gemma3nModel
from fairseq2.models.gemma3n.projection import SoftcappedProjection
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerFrontend,
    create_default_sdpa,
)
from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork


class TanhGELU(nn.Module):
    """GELU activation with tanh approximation.

    Uses tanh approximation to match HuggingFace Gemma3n implementation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


from fairseq2.nn import (
    Embedding,
    PositionEncoder,
    Projection,
    RMSNorm,
    StandardEmbedding,
    TiedProjection,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from fairseq2.nn.projection import Linear


def create_gemma3n_model(
    config: Gemma3nConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Gemma3nModel:
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

    def create_model(self) -> Gemma3nModel:
        """Create the full Gemma3n model."""
        embed = self.create_embedding()
        audio_tower = self.create_audio_tower()
        decoder_frontend = self.create_decoder_frontend(embed, audio_tower)
        decoder = self.create_decoder()
        final_proj = self.create_final_projection(embed)

        return Gemma3nModel(
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

    def create_audio_tower(self) -> "Gemma3nAudioTower | None":
        """Create audio tower if config has audio_config."""
        if self._config.audio_config is None:
            return None

        from fairseq2.models.gemma3n.audio.tower import Gemma3nAudioTower

        return Gemma3nAudioTower(
            audio_config=self._config.audio_config,
            text_config=self._config,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder_frontend(
        self, embed: Embedding, audio_tower: "Gemma3nAudioTower | None"
    ) -> TransformerFrontend:
        """Create the decoder frontend with PLE support."""
        # PLE normalization
        ple_norm = RMSNorm(
            self._config.ple_hidden_dim,
            bias=False,
            eps=self._config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

        return Gemma3nFrontend(
            model_dim=self._config.model_dim,
            embed=embed,
            pos_encoder=None,
            vocab_size_per_layer=self._config.vocab_size_per_layer,
            num_layers=self._config.num_layers,
            ple_hidden_dim=self._config.ple_hidden_dim,
            ple_norm=ple_norm,
            no_scale=False,  # Gemma3n scales embeddings by sqrt(model_dim)
            dropout_p=0.0,
            audio_tower=audio_tower,
            audio_token_id=(
                self._config.audio_config.vocab_offset
                if self._config.audio_config
                else None
            ),
            num_audio_tokens=self._config.num_audio_tokens,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder(self) -> Gemma3nDecoder:
        """Create the Gemma3n decoder with AltUp and PLE."""
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

        return Gemma3nDecoder(
            layers=layers,
            layer_norm=layer_norm,
            model_dim=self._config.model_dim,
            num_altup_inputs=self._config.altup_num_inputs,
            device=self._device,
            dtype=self._dtype,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        """Create the final output projection with optional softcapping."""
        if not isinstance(embed, StandardEmbedding):
            raise TypeError(
                f"`embed` must be `StandardEmbedding`, got `{type(embed)}` instead."
            )

        base_proj = TiedProjection(embed.weight, bias=None)

        # Wrap with softcapping if configured
        if self._config.final_logit_softcapping is not None:
            return SoftcappedProjection(base_proj, self._config.final_logit_softcapping)

        return base_proj


def create_gemma3n_decoder_layer(
    layer_idx: int,
    config: Gemma3nConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Gemma3nDecoderLayer:
    """Create a Gemma3n decoder layer with AltUp, LAuReL, and PLE.

    :param layer_idx: The layer index (0-based).
    :param config: The Gemma3n configuration.
    :returns: A configured Gemma3n decoder layer.
    """
    is_global = is_global_layer(layer_idx, config.num_layers)

    # KV projection sharing role
    kv_projection_role = get_kv_projection_role(
        layer_idx, is_global, config.num_layers, config.num_kv_shared_layers
    )

    # Position encoder (different theta for local vs global)
    if is_global:
        # Global layers: use larger theta for longer-range dependencies
        pos_encoder = ReferenceRotaryEncoder(
            encoding_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta_global,
            device=device,
        )
    else:
        # Local layers: use standard theta
        pos_encoder = ReferenceRotaryEncoder(
            encoding_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            device=device,
        )

    # Use scale=1.0 to disable attention logit scaling because Gemma3n uses QK normalization
    if is_global:
        # Global layers: full causal attention
        attention_bias = CausalAttentionBias()
    else:
        # Local layers: sliding window attention
        attention_bias = CausalAttentionBias(attn_window_len=config.sliding_window)

    sdpa = create_default_sdpa(
        attention_bias,
        dropout_p=0.0,
        scale=1.0,  # Disable scaling - Gemma3n uses QK normalization instead
    )

    # QKV normalization
    q_norm = RMSNorm(
        config.head_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    k_norm = RMSNorm(
        config.head_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    v_norm = RMSNorm(
        config.head_dim,
        bias=False,
        eps=config.rms_norm_eps,
        elementwise_affine=False,
        device=device,
        dtype=dtype,
    )

    # Self-attention
    self_attn = StandardMultiheadAttention(
        model_dim=config.model_dim,
        num_heads=config.num_attn_heads,
        num_key_value_heads=config.num_key_value_heads,
        sdpa=sdpa,
        pos_encoder=pos_encoder,
        q_norm=q_norm,
        k_norm=k_norm,
        v_norm=v_norm,
        bias=False,
        device=device,
        dtype=dtype,
    )

    # FFN (global vs local)
    # Sparsity pattern: first 10 layers have 0.95 sparsity regardless of type
    num_sparse_layers = 10 if config.num_layers > 10 else 0
    activation_sparsity = 0.95 if layer_idx < num_sparse_layers else 0.0

    ffn: FeedForwardNetwork
    if is_global:
        ffn = GLUFeedForwardNetwork(
            model_dim=config.model_dim,
            inner_dim=config.ffn_inner_dim,
            bias=False,
            gate_activation=TanhGELU(),  # Use tanh-approximated GELU
            inner_dim_scale=1.0,  # Disable 2/3 scaling for Gemma3n
            activation_sparsity=activation_sparsity,
            device=device,
            dtype=dtype,
        )
    else:
        # Local layers: use AltUp FFN with activation sparsity
        ffn = AltUpFeedForwardNetwork(
            model_dim=config.model_dim,
            inner_dim=config.altup_hidden_dim,
            bias=False,
            activation_sparsity=activation_sparsity,
            device=device,
            dtype=dtype,
        )

    # Normalizations
    input_layernorm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    post_attention_layernorm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    pre_feedforward_layernorm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    post_feedforward_layernorm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )

    # LAuReL
    post_laurel_norm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    laurel = Gemma3nLAuReL(
        model_dim=config.model_dim,
        rank=config.laurel_rank,
        layer_norm=post_laurel_norm,
        device=device,
        dtype=dtype,
    )

    # AltUp router normalization
    router_norm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )
    altup = Gemma3nAltUp(
        model_dim=config.model_dim,
        num_inputs=config.altup_num_inputs,
        active_idx=config.altup_active_idx,
        router_norm=router_norm,
        coef_clip=config.altup_coef_clip,
        device=device,
        dtype=dtype,
    )

    # PLE components (per-layer gating and projection)
    per_layer_input_gate = Linear(
        config.model_dim,
        config.ple_hidden_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    per_layer_projection = Linear(
        config.ple_hidden_dim,
        config.model_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    post_per_layer_input_norm = RMSNorm(
        config.model_dim, bias=False, eps=config.rms_norm_eps, device=device, dtype=dtype
    )

    # Hidden activation for PLE (tanh-approximated GELU to match HF)
    hidden_activation = TanhGELU()

    return Gemma3nDecoderLayer(
        self_attn=self_attn,
        ffn=ffn,
        layer_idx=layer_idx,
        is_global=is_global,
        kv_projection_role=kv_projection_role,
        input_layernorm=input_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        pre_feedforward_layernorm=pre_feedforward_layernorm,
        post_feedforward_layernorm=post_feedforward_layernorm,
        laurel=laurel,
        altup=altup,
        altup_active_idx=config.altup_active_idx,
        altup_correct_scale=config.altup_correct_scale,
        per_layer_input_gate=per_layer_input_gate,
        per_layer_projection=per_layer_projection,
        post_per_layer_input_norm=post_per_layer_input_norm,
        hidden_activation=hidden_activation,
        dropout_p=0.0,
        device=device,
        dtype=dtype,
    )

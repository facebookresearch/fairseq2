# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory for building Gemma 4 models from :class:`Gemma4Config`.

Follows the pattern established by :mod:`fairseq2.models.gemma3n.factory`.
The factory assembles a complete model (embedding, frontend, decoder,
projection) with optional audio tower for multimodal support.

The component classes live in their own modules:

* :class:`Gemma4Model` -- :mod:`fairseq2.models.gemma4.model`
* :class:`Gemma4Decoder` -- :mod:`fairseq2.models.gemma4.decoder`
* :class:`Gemma4Frontend` -- :mod:`fairseq2.models.gemma4.frontend`
* :class:`Gemma4DecoderLayer` -- :mod:`fairseq2.models.gemma4.decoder_layer`
* :class:`Gemma4Attention` -- :mod:`fairseq2.models.gemma4.attention`
"""

from __future__ import annotations

import torch
from torch.nn import Module

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.gemma3n.kv_projection import KVProjectionRole
from fairseq2.models.gemma3n.projection import SoftcappedProjection
from fairseq2.models.gemma4.attention import (
    Gemma4Attention,
    Gemma4ProportionalRotaryEncoder,
)
from fairseq2.models.gemma4.config import Gemma4Config, get_kv_projection_role
from fairseq2.models.gemma4.decoder import Gemma4Decoder
from fairseq2.models.gemma4.decoder_layer import Gemma4DecoderLayer
from fairseq2.models.gemma4.frontend import Gemma4Frontend
from fairseq2.models.gemma4.model import Gemma4Model
from fairseq2.models.gemma4.moe import Gemma4Experts, Gemma4Router
from fairseq2.models.gemma4.sdpa import Gemma4SDPA
from fairseq2.models.transformer import (
    CausalAttentionBias,
    GLUFeedForwardNetwork,
)
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Projection,
    RMSNorm,
    StandardEmbedding,
    TiedProjection,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from fairseq2.nn.projection import Linear

__all__ = ["Gemma4Factory", "create_gemma4_model"]


def create_gemma4_model(
    config: Gemma4Config,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Gemma4Model:
    """Create a Gemma 4 language model.

    :param config: The Gemma 4 configuration.
    :param device: The device on which to initialise the model.
    :param dtype: The data type of the model parameters and buffers.
    :returns: A Gemma 4 model.
    """
    gangs = maybe_get_current_gangs()

    return Gemma4Factory(config, device=device, dtype=dtype, gangs=gangs).create_model()


class Gemma4Factory:
    """Factory for creating Gemma 4 model components."""

    _config: Gemma4Config
    _device: Device | None
    _dtype: DataType | None
    _gangs: Gangs | None

    def __init__(
        self,
        config: Gemma4Config,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        gangs: Gangs | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._dtype = dtype
        self._gangs = gangs

    def create_model(self) -> Gemma4Model:
        """Create the full Gemma 4 model."""
        embed = self.create_embedding()
        frontend = self.create_decoder_frontend(embed)
        decoder = self.create_decoder()
        final_proj = self.create_final_projection(embed)
        audio_tower = self.create_audio_tower()
        audio_embedder = self.create_audio_embedder()

        return Gemma4Model(
            self._config.model_dim,
            frontend,
            decoder,
            final_proj,
            self._config.pad_idx,
            self._config.max_seq_len,
            audio_tower=audio_tower,
            audio_embedder=audio_embedder,
        )

    def create_embedding(self) -> Embedding:
        """Create the token embedding layer."""
        config = self._config

        return StandardEmbedding(
            config.vocab_size,
            config.model_dim,
            config.pad_idx,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder_frontend(self, embed: Embedding) -> Gemma4Frontend:
        """Create the decoder frontend with optional PLE and audio injection."""
        config = self._config

        ple_norm: LayerNorm | None = None
        if config.has_ple:
            ple_norm = RMSNorm(
                config.ple_hidden_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )

        # Enable audio injection if audio tower is configured.
        audio_token_id: int | None = None
        if config.audio_config is not None:
            audio_token_id = config.audio_token_id

        return Gemma4Frontend(
            model_dim=config.model_dim,
            embed=embed,
            num_layers=config.num_layers,
            ple_hidden_dim=config.ple_hidden_dim,
            vocab_size_per_layer_input=config.vocab_size_per_layer_input,
            ple_norm=ple_norm,
            audio_token_id=audio_token_id,
            pad_idx=config.pad_idx,
            device=self._device,
            dtype=self._dtype,
        )

    def create_decoder(self) -> Gemma4Decoder:
        """Create the Gemma 4 decoder stack."""
        config = self._config

        layer_types_list = config.layer_types
        layers: list[Gemma4DecoderLayer] = []
        layer_kv_roles: list[KVProjectionRole] = []

        for i in range(config.num_layers):
            layer_type = layer_types_list[i]
            is_full = layer_type == "full_attention"

            kv_role = get_kv_projection_role(
                i,
                layer_type,
                config.num_layers,
                config.num_kv_shared_layers,
                layer_types_list,
            )

            layer = self.create_decoder_layer(i, layer_type, is_full, kv_role)
            layers.append(layer)
            layer_kv_roles.append(kv_role)

        layer_norm = RMSNorm(
            config.model_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

        return Gemma4Decoder(
            layers=layers,
            layer_norm=layer_norm,
            layer_kv_roles=layer_kv_roles,
            layer_types=layer_types_list,
        )

    def create_decoder_layer(
        self,
        layer_idx: int,
        layer_type: str,
        is_full: bool,
        kv_role: KVProjectionRole,
    ) -> Gemma4DecoderLayer:
        """Create a single Gemma 4 decoder layer.

        :param layer_idx: Zero-based layer index.
        :param layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        :param is_full: Whether this is a full (global) attention layer.
        :param kv_role: KV projection sharing role for this layer.
        :returns: A configured decoder layer.
        """
        config = self._config

        # --- Attention ---
        self_attn = self._create_attention(layer_idx, layer_type, is_full, kv_role)

        # --- FFN ---
        ffn = self._create_ffn(layer_idx, layer_type, kv_role)

        # --- Layer norms (4 core norms) ---
        input_layernorm = RMSNorm(
            config.model_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )
        post_attention_layernorm = RMSNorm(
            config.model_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )
        pre_feedforward_layernorm = RMSNorm(
            config.model_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )
        post_feedforward_layernorm = RMSNorm(
            config.model_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

        # --- Optional PLE ---
        per_layer_input_gate: Linear | None = None
        per_layer_projection: Linear | None = None
        post_per_layer_input_norm: LayerNorm | None = None

        if config.has_ple:
            per_layer_input_gate = Linear(
                config.model_dim,
                config.ple_hidden_dim,
                bias=False,
                device=self._device,
                dtype=self._dtype,
            )
            per_layer_projection = Linear(
                config.ple_hidden_dim,
                config.model_dim,
                bias=False,
                device=self._device,
                dtype=self._dtype,
            )
            post_per_layer_input_norm = RMSNorm(
                config.model_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )

        # --- Optional MoE ---
        router: Module | None = None
        experts: Module | None = None
        post_feedforward_layernorm_1: LayerNorm | None = None
        pre_feedforward_layernorm_2: LayerNorm | None = None
        post_feedforward_layernorm_2: LayerNorm | None = None

        if config.enable_moe:
            assert config.num_experts is not None
            assert config.top_k_experts is not None
            assert config.moe_intermediate_size is not None

            router = Gemma4Router(
                config.model_dim,
                config.num_experts,
                config.top_k_experts,
                rms_norm_eps=config.rms_norm_eps,
            )
            experts = Gemma4Experts(
                config.model_dim,
                config.num_experts,
                config.moe_intermediate_size,
            )
            post_feedforward_layernorm_1 = RMSNorm(
                config.model_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )
            pre_feedforward_layernorm_2 = RMSNorm(
                config.model_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )
            post_feedforward_layernorm_2 = RMSNorm(
                config.model_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )

        return Gemma4DecoderLayer(
            self_attn=self_attn,
            ffn=ffn,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            pre_feedforward_layernorm=pre_feedforward_layernorm,
            post_feedforward_layernorm=post_feedforward_layernorm,
            per_layer_input_gate=per_layer_input_gate,
            per_layer_projection=per_layer_projection,
            post_per_layer_input_norm=post_per_layer_input_norm,
            router=router,
            experts=experts,
            post_feedforward_layernorm_1=post_feedforward_layernorm_1,
            pre_feedforward_layernorm_2=pre_feedforward_layernorm_2,
            post_feedforward_layernorm_2=post_feedforward_layernorm_2,
            activation_fn=config.hidden_activation,
        )

    def _create_attention(
        self,
        layer_idx: int,
        layer_type: str,
        is_full: bool,
        kv_role: KVProjectionRole,
    ) -> Gemma4Attention:
        """Create the multi-head attention module for a decoder layer.

        :param layer_idx: Zero-based layer index.
        :param layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        :param is_full: Whether this is a full (global) attention layer.
        :param kv_role: KV projection sharing role.
        :returns: A configured :class:`Gemma4Attention`.
        """
        config = self._config

        if is_full:
            # Full (global) attention: large head_dim, partial RoPE.
            head_dim = config.global_head_dim
            num_kv_heads = (
                config.num_global_key_value_heads
                if config.num_global_key_value_heads is not None
                else config.num_key_value_heads
            )
            rope_theta = config.rope_theta_global
            k_eq_v = config.attention_k_eq_v
        else:
            # Sliding (local) attention: standard head_dim, full RoPE.
            head_dim = config.head_dim
            num_kv_heads = config.num_key_value_heads
            rope_theta = config.rope_theta
            k_eq_v = False

        is_consumer = kv_role == KVProjectionRole.CONSUMER

        # Position encoder (RoPE).
        pos_encoder: ReferenceRotaryEncoder
        if is_full and config.partial_rotary_factor < 1.0:
            # Proportional RoPE: zero-padded inv_freq over full head_dim.
            # rotate_half pairs dim_i with dim_{i + head_dim//2}, matching
            # HuggingFace's implementation exactly.
            rope_dim = int(head_dim * config.partial_rotary_factor)
            pos_encoder = Gemma4ProportionalRotaryEncoder(
                head_dim=head_dim,
                rope_dim=rope_dim,
                max_seq_len=config.max_seq_len,
                theta=rope_theta,
                device=self._device,
            )
        else:
            # Full rotation (sliding layers, or full layers without partial).
            pos_encoder = ReferenceRotaryEncoder(
                encoding_dim=head_dim,
                max_seq_len=config.max_seq_len,
                theta=rope_theta,
                device=self._device,
            )

        # Attention bias.
        if is_full:
            attn_bias = CausalAttentionBias()
        else:
            attn_bias = CausalAttentionBias(attn_window_len=config.sliding_window)

        # Gemma 4 uses QK-norm so we disable SDPA scaling.
        # Use Gemma4SDPA which passes scale directly to the PyTorch kernel
        # instead of pre-scaling Q (avoids bfloat16 precision loss in MoE).
        sdpa = Gemma4SDPA(attn_bias, dropout_p=0.0, scale=1.0)

        # Q norm (always present, even for consumer layers).
        q_norm = RMSNorm(
            head_dim,
            bias=False,
            eps=config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

        # K norm: only for non-consumer layers.  Consumer layers receive
        # pre-computed K/V from the SOURCE layer (already normed and RoPE'd).
        k_norm: LayerNorm | None = None
        if not is_consumer:
            k_norm = RMSNorm(
                head_dim,
                bias=False,
                eps=config.rms_norm_eps,
                device=self._device,
                dtype=self._dtype,
            )

        # V norm: RMSNorm WITHOUT learnable scale (elementwise_affine=False).
        # Only for non-consumer layers (consumer layers use SOURCE's V).
        v_norm: LayerNorm | None = None
        if not is_consumer:
            v_norm = RMSNorm(
                head_dim,
                bias=False,
                eps=config.rms_norm_eps,
                elementwise_affine=False,
                device=self._device,
                dtype=self._dtype,
            )

        return Gemma4Attention(
            model_dim=config.model_dim,
            num_heads=config.num_attn_heads,
            sdpa=sdpa,
            head_dim=head_dim,
            num_key_value_heads=num_kv_heads,
            pos_encoder=pos_encoder,
            q_norm=q_norm,
            k_norm=k_norm,
            v_norm=v_norm,
            k_eq_v=k_eq_v,
            is_kv_consumer=is_consumer,
        )

    def _create_ffn(
        self,
        layer_idx: int,
        layer_type: str,
        kv_role: KVProjectionRole,
    ) -> GLUFeedForwardNetwork:
        """Create the feed-forward network for a decoder layer.

        :param layer_idx: Zero-based layer index.
        :param layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        :param kv_role: KV projection sharing role (CONSUMER layers may use
            double-wide MLP when ``use_double_wide_mlp`` is set).
        :returns: A :class:`GLUFeedForwardNetwork`.
        """
        config = self._config

        inner_dim = config.ffn_inner_dim

        # KV-shared (CONSUMER) layers may use 2x intermediate size.
        if config.use_double_wide_mlp and kv_role == KVProjectionRole.CONSUMER:
            inner_dim *= 2

        return GLUFeedForwardNetwork(
            model_dim=config.model_dim,
            inner_dim=inner_dim,
            bias=False,
            gate_activation=torch.nn.GELU(approximate="tanh"),
            inner_dim_scale=1.0,  # Disable the default 2/3 scaling.
            device=self._device,
            dtype=self._dtype,
        )

    def create_final_projection(self, embed: Embedding) -> Projection:
        """Create the final output projection with optional softcapping.

        :param embed: The token embedding (used for weight tying).
        :returns: A projection, optionally wrapped with
            :class:`SoftcappedProjection`.
        """
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, StandardEmbedding):
                raise TypeError(
                    f"`embed` must be `StandardEmbedding` when "
                    f"`tied_embeddings=True`, got `{type(embed)}` instead."
                )
            base_proj: Projection = TiedProjection(embed.weight, bias=None)
        else:
            base_proj = Linear(
                config.model_dim,
                config.vocab_size,
                bias=False,
                device=self._device,
                dtype=self._dtype,
            )

        if config.final_logit_soft_cap is not None:
            return SoftcappedProjection(base_proj, config.final_logit_soft_cap)

        return base_proj

    def create_audio_tower(self) -> Module | None:
        """Create the audio tower for mel-spectrogram encoding.

        :returns: A :class:`Gemma4AudioTower` if audio is configured AND the
            audio_mode is ``"conformer"``; ``None`` otherwise.

        The Gemma 4 Unified family (``audio_mode="linear"``) has no audio
        tower — raw waveform frames are fed directly through the
        multimodal embedder (see :meth:`create_audio_embedder` and
        ``Gemma4Model.forward``).
        """
        config = self._config

        if config.audio_config is None:
            return None

        # Unified family: no tower; embedder consumes raw waveform frames.
        if getattr(config.audio_config, "audio_mode", "conformer") == "linear":
            return None

        from fairseq2.models.gemma4.audio.tower import Gemma4AudioTower

        return Gemma4AudioTower(
            audio_config=config.audio_config,
            device=self._device,
            dtype=self._dtype,
        )

    def create_audio_embedder(self) -> Module | None:
        """Create the audio embedder to project audio features to text space.

        :returns: A :class:`Gemma4MultimodalAudioEmbedder` if audio is
            configured, ``None`` otherwise.
        """
        config = self._config

        if config.audio_config is None:
            return None

        from fairseq2.models.gemma4.audio.embedder import (
            Gemma4MultimodalAudioEmbedder,
        )

        return Gemma4MultimodalAudioEmbedder(
            output_proj_dims=config.audio_config.output_proj_dims,
            text_model_dim=config.model_dim,
            rms_norm_eps=config.audio_config.rms_norm_eps,
            device=self._device,
            dtype=self._dtype,
        )

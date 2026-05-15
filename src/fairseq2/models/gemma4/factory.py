# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory for building Gemma 4 models from :class:`Gemma4Config`.

Follows the pattern established by :mod:`fairseq2.models.gemma3n.factory`
but simplified: no AltUp, no LAuReL, no audio tower.  The factory
assembles a complete model (embedding, frontend, decoder, projection)
using the existing :class:`Gemma4DecoderLayer` and :class:`Gemma4Attention`
modules.

This file also defines three supporting classes that are tightly coupled
to the factory and not large enough to warrant separate files:

* :class:`Gemma4Model` -- top-level decoder-only LM.
* :class:`Gemma4Decoder` -- decoder stack (layers + final norm + KV sharing).
* :class:`Gemma4Frontend` -- embedding + optional PLE.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, final, overload

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs, maybe_get_current_gangs
from fairseq2.models.clm import CausalLM
from fairseq2.models.gemma3n.kv_projection import (
    KVProjectionRole,
    KVProjectionType,
)
from fairseq2.models.gemma3n.projection import SoftcappedProjection
from fairseq2.models.gemma4.attention import Gemma4Attention
from fairseq2.models.gemma4.config import Gemma4Config, get_kv_projection_role
from fairseq2.models.gemma4.decoder_layer import Gemma4DecoderLayer
from fairseq2.models.gemma4.moe import Gemma4Experts, Gemma4Router
from fairseq2.models.gemma4.sdpa import Gemma4SDPA
from fairseq2.models.transformer import (
    AttentionBiasCache,
    CausalAttentionBias,
    GLUFeedForwardNetwork,
    create_default_sdpa,
)
from fairseq2.nn import (
    BatchLayout,
    Embedding,
    IncrementalStateBag,
    LayerNorm,
    Projection,
    RMSNorm,
    StandardEmbedding,
    TiedProjection,
)
from fairseq2.nn.functional import cross_entropy
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from fairseq2.nn.projection import Linear

from fairseq2.models.gemma4.attention import Gemma4ProportionalRotaryEncoder


# ---------------------------------------------------------------------------
# Gemma4Model
# ---------------------------------------------------------------------------

@final
class Gemma4Model(CausalLM):
    """Gemma 4 decoder-only causal language model."""

    model_dim: int
    decoder_frontend: Gemma4Frontend
    decoder: Gemma4Decoder
    final_proj: Projection
    pad_idx: int | None

    def __init__(
        self,
        model_dim: int,
        decoder_frontend: Gemma4Frontend,
        decoder: Gemma4Decoder,
        final_proj: Projection,
        pad_idx: int | None,
        max_seq_len: int,
    ) -> None:
        """
        :param model_dim: The model dimensionality.
        :param decoder_frontend: The decoder frontend (embedding + optional PLE).
        :param decoder: The decoder stack.
        :param final_proj: The projection to apply to decoder outputs.
        :param pad_idx: The index of the pad symbol in the vocabulary.
        :param max_seq_len: The maximum sequence length.
        """
        super().__init__(max_seq_len)

        self.model_dim = model_dim
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pad_idx = pad_idx

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        :param seqs: Input token IDs. *Shape:* ``(B, S)``.
        :param seqs_layout: Layout information.
        :param targets: Target token IDs for loss computation.
        :param state_bag: Incremental decoding state.
        :param label_smoothing: Label smoothing factor.
        :param target_mask: Mask for targets.
        :param reduction: Loss reduction method.
        :param return_logits: If True, return both loss and logits.
        :returns: Logits or loss (or both if return_logits=True).
        """
        seqs, seqs_layout, per_layer_embeds = self.decoder_frontend(
            seqs,
            seqs_layout,
            state_bag=state_bag,
        )

        decoder_output = self.decoder(
            seqs,
            seqs_layout,
            state_bag=state_bag,
            per_layer_embeds=per_layer_embeds,
        )

        del seqs

        if targets is None:
            return self.final_proj(decoder_output)

        if not return_logits:
            return self.compute_fused_loss(
                decoder_output,
                targets,
                label_smoothing=label_smoothing,
                target_mask=target_mask,
                reduction=reduction,
            )

        logits = self.final_proj(decoder_output)

        del decoder_output

        loss = self.compute_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

        return loss, logits

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compute_fused_loss(
        self,
        decoder_output: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        logits = self.final_proj(decoder_output)

        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compile_loss(self, *args: Any, **kwargs: Any) -> None:
        self.compute_fused_loss = torch.compile(  # type: ignore[method-assign]
            self.compute_fused_loss, *args, **kwargs
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"pad_idx={self.pad_idx}, "
            f"max_seq_len={self.max_seq_len}"
        )


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

    return Gemma4Factory(
        config, device=device, dtype=dtype, gangs=gangs
    ).create_model()


# ---------------------------------------------------------------------------
# Gemma4Decoder -- Gemma 4 decoder without AltUp
# ---------------------------------------------------------------------------

class Gemma4Decoder(Module):
    """Gemma 4 decoder stack with KV-sharing support.

    Unlike :class:`~fairseq2.models.gemma3n.decoder.Gemma3nDecoder`, this
    decoder has **no** AltUp projections.  Input and output are plain 3-D
    tensors ``(B, S, M)``.
    """

    layers: ModuleList
    layer_norm: LayerNorm
    _layer_kv_roles: list[KVProjectionRole]
    _layer_types: list[str]
    _has_kv_projection_sharing: bool

    def __init__(
        self,
        layers: Sequence[Gemma4DecoderLayer],
        layer_norm: LayerNorm,
        *,
        layer_kv_roles: Sequence[KVProjectionRole],
        layer_types: Sequence[str],
    ) -> None:
        """
        :param layers: Ordered sequence of :class:`Gemma4DecoderLayer`.
        :param layer_norm: Final RMSNorm applied after the last layer.
        :param layer_kv_roles: Per-layer :class:`KVProjectionRole`.
        :param layer_types: Per-layer attention type strings
            (``"sliding_attention"`` or ``"full_attention"``).
        """
        super().__init__()

        self.layers = ModuleList(layers)
        self.layer_norm = layer_norm

        self._layer_kv_roles = list(layer_kv_roles)
        self._layer_types = list(layer_types)

        self._has_kv_projection_sharing = any(
            role != KVProjectionRole.NONE for role in self._layer_kv_roles
        )

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        per_layer_embeds: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs: Hidden states. *Shape:* ``(B, S, M)``.
        :param seqs_layout: Batch layout for attention masking.
        :param state_bag: Incremental state bag for KV-cache.
        :param per_layer_embeds: PLE embeddings. *Shape:*
            ``(B, S, num_layers, ple_dim)``.  ``None`` when PLE is disabled.
        :returns: Decoder output. *Shape:* ``(B, S, M)``.
        """
        # Prepare KV sharing slots.
        kv_slots: dict[KVProjectionType, tuple[Tensor, Tensor] | None] | None = None
        if self._has_kv_projection_sharing:
            kv_slots = {
                KVProjectionType.LOCAL: None,
                KVProjectionType.GLOBAL: None,
            }

        attn_bias_cache = AttentionBiasCache()

        for layer_idx, layer in enumerate(self.layers):
            # Per-layer embedding slice.
            layer_ple: Tensor | None = None
            if per_layer_embeds is not None:
                layer_ple = per_layer_embeds[..., layer_idx, :]

            # Resolve KV sharing arguments.
            pre_computed_kv: tuple[Tensor, Tensor] | None = None
            kv_storage_callback: Callable[[Tensor, Tensor], None] | None = None

            if kv_slots is not None:
                role = self._layer_kv_roles[layer_idx]
                layer_type = self._layer_types[layer_idx]
                slot_key = (
                    KVProjectionType.GLOBAL
                    if layer_type == "full_attention"
                    else KVProjectionType.LOCAL
                )

                if role == KVProjectionRole.CONSUMER:
                    pre_computed_kv = kv_slots[slot_key]
                    if pre_computed_kv is None:
                        raise RuntimeError(
                            f"Layer {layer_idx} ({slot_key.value}) is a CONSUMER "
                            f"but no SOURCE has populated the {slot_key.value} slot."
                        )
                elif role == KVProjectionRole.SOURCE:

                    def _make_cb(
                        s: dict[KVProjectionType, tuple[Tensor, Tensor] | None],
                        k: KVProjectionType,
                    ) -> Callable[[Tensor, Tensor], None]:
                        def cb(key: Tensor, val: Tensor) -> None:
                            s[k] = (key, val)

                        return cb

                    kv_storage_callback = _make_cb(kv_slots, slot_key)

            # Forward through the decoder layer.
            seqs = layer(
                seqs,
                seqs_layout,
                attn_bias_cache,
                per_layer_input=layer_ple,
                state_bag=state_bag,
                pre_computed_kv=pre_computed_kv,
                kv_storage_callback=kv_storage_callback,
            )

        seqs = self.layer_norm(seqs)

        return seqs

    def compile_layerwise(self, *args: Any, **kwargs: Any) -> None:
        """Compile each layer individually."""
        for layer in self.layers:
            layer.compile(*args, **kwargs)

        if self.layer_norm is not None:
            self.layer_norm.compile(*args, **kwargs)

    if TYPE_CHECKING:
        __call__ = forward


# ---------------------------------------------------------------------------
# Gemma4Frontend -- PLE-aware frontend (wraps Gemma3nFrontend)
# ---------------------------------------------------------------------------

class Gemma4Frontend(Module):
    """Gemma 4 decoder frontend with optional Per-Layer Embeddings (PLE).

    When PLE is disabled (``ple_hidden_dim == 0``), this is a simple embedding
    lookup with ``sqrt(model_dim)`` scaling.  When PLE is enabled, it behaves
    identically to :class:`~fairseq2.models.gemma3n.frontend.Gemma3nFrontend`
    (discrete + continuous per-layer embeddings).
    """

    embed: Embedding
    scale: float

    # PLE modules (None when PLE is disabled).
    embed_tokens_per_layer: StandardEmbedding | None
    per_layer_model_projection: Linear | None
    per_layer_projection_norm: LayerNorm | None
    num_layers: int
    ple_hidden_dim: int

    def __init__(
        self,
        model_dim: int,
        embed: Embedding,
        *,
        num_layers: int,
        ple_hidden_dim: int = 0,
        vocab_size_per_layer_input: int = 0,
        ple_norm: LayerNorm | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Model dimensionality.
        :param embed: Token embedding table.
        :param num_layers: Number of decoder layers.
        :param ple_hidden_dim: Hidden dim for PLE.  0 disables PLE.
        :param vocab_size_per_layer_input: Vocabulary size for PLE lookup.
        :param ple_norm: RMSNorm for PLE projection (required when PLE enabled).
        :param device: Device.
        :param dtype: Data type.
        """
        super().__init__()

        self.embed = embed
        self.scale = model_dim ** 0.5
        self.num_layers = num_layers
        self.ple_hidden_dim = ple_hidden_dim

        if ple_hidden_dim > 0 and vocab_size_per_layer_input > 0:
            # PLE enabled.
            self.embed_tokens_per_layer = StandardEmbedding(
                num_embeddings=vocab_size_per_layer_input,
                embed_dim=num_layers * ple_hidden_dim,
                pad_idx=None,
                device=device,
                dtype=dtype,
            )

            self.per_layer_model_projection = Linear(
                model_dim,
                num_layers * ple_hidden_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )

            if ple_norm is None:
                raise ValueError(
                    "`ple_norm` must be provided when PLE is enabled."
                )
            self.per_layer_projection_norm = ple_norm

            # Scaling buffers (non-persistent).
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(model_dim ** -0.5, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_embed_scale",
                torch.tensor(ple_hidden_dim ** 0.5, device=device, dtype=dtype),
                persistent=False,
            )
        else:
            # PLE disabled.
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        audio_embeds: Tensor | None = None,
        vision_features: Tensor | None = None,
    ) -> tuple[Tensor, BatchLayout, Tensor | None]:
        """
        :param seqs: Token IDs. *Shape:* ``(B, S)``.
        :param seqs_layout: Layout information.
        :param state_bag: Incremental decoding state.
        :param audio_embeds: Unused (Gemma 4 has no audio tower).
        :param vision_features: Unused (Gemma 4 has no vision tower).
        :returns:
            - Embeddings ``(B, S, M)``
            - Layout
            - Per-layer embeddings ``(B, S, L, ple_dim)`` or ``None``
        """
        token_ids = seqs

        seqs = self.embed(seqs)
        seqs = seqs * self.scale

        if self.embed_tokens_per_layer is not None:
            per_layer_inputs = self._compute_ple(token_ids, seqs)
        else:
            per_layer_inputs = None

        return seqs, seqs_layout, per_layer_inputs

    def _compute_ple(
        self, token_ids: Tensor, seqs: Tensor
    ) -> Tensor:
        """Compute per-layer embeddings (discrete + continuous).

        :param token_ids: Token IDs ``(B, S)``.
        :param seqs: Scaled embeddings ``(B, S, M)``.
        :returns: PLE ``(B, S, num_layers, ple_hidden_dim)``.
        """
        assert self.embed_tokens_per_layer is not None
        assert self.per_layer_model_projection is not None
        assert self.per_layer_projection_norm is not None

        # Discrete PLE.
        ple_token_ids = torch.clamp(
            token_ids, max=self.embed_tokens_per_layer.num_embeddings - 1
        )
        discrete = self.embed_tokens_per_layer(ple_token_ids)  # (B, S, L*P)
        discrete = discrete * self.per_layer_embed_scale  # type: ignore[operator]
        discrete = discrete.reshape(
            *token_ids.shape, self.num_layers, self.ple_hidden_dim
        )

        # Continuous PLE.
        continuous = self.per_layer_model_projection(seqs)  # (B, S, L*P)
        continuous = continuous * self.per_layer_projection_scale  # type: ignore[operator]
        continuous = continuous.reshape(
            *seqs.shape[:-1], self.num_layers, self.ple_hidden_dim
        )
        continuous = self.per_layer_projection_norm(continuous)

        # Combine.
        scale = self.per_layer_input_scale  # type: ignore[assignment]
        return (continuous + discrete) * scale

    if TYPE_CHECKING:
        __call__ = forward


# ---------------------------------------------------------------------------
# Gemma4Factory
# ---------------------------------------------------------------------------

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

        return Gemma4Model(
            self._config.model_dim,
            frontend,
            decoder,
            final_proj,
            self._config.pad_idx,
            self._config.max_seq_len,
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
        """Create the decoder frontend with optional PLE."""
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

        return Gemma4Frontend(
            model_dim=config.model_dim,
            embed=embed,
            num_layers=config.num_layers,
            ple_hidden_dim=config.ple_hidden_dim,
            vocab_size_per_layer_input=config.vocab_size_per_layer_input,
            ple_norm=ple_norm,
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
            encoding_dim = int(head_dim * config.partial_rotary_factor)
            rope_theta = config.rope_theta_global
            k_eq_v = config.attention_k_eq_v
        else:
            # Sliding (local) attention: standard head_dim, full RoPE.
            head_dim = config.head_dim
            num_kv_heads = config.num_key_value_heads
            encoding_dim = head_dim  # Full rotation.
            rope_theta = config.rope_theta
            k_eq_v = False

        is_consumer = kv_role == KVProjectionRole.CONSUMER

        # Position encoder (RoPE).
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

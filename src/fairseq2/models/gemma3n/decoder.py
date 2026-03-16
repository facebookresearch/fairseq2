# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, final

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma3n.decoder_layer import Gemma3nDecoderLayer
from fairseq2.models.gemma3n.kv_projection import KVProjectionRole, KVProjectionType
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm
from fairseq2.nn.projection import Linear


class Gemma3nDecoderBase(Module, ABC):
    """Base class for Gemma3n decoders with per-layer embedding support."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        per_layer_embeds: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs: Input embeddings. *Shape:* :math:`(N,S,M)`.
        :param seqs_layout: Layout information for ``seqs``.
        :param state_bag: Incremental decoding state (KV cache).
        :param per_layer_embeds: Per-layer embeddings. *Shape:* :math:`(N,S,L,H)`.
        :returns: Decoder output. *Shape:* :math:`(N,S,M)`.
        """

    if TYPE_CHECKING:
        __call__ = forward

    @abstractmethod
    def compile_layerwise(self, *args: Any, **kwargs: Any) -> None: ...


@final
class Gemma3nDecoder(Gemma3nDecoderBase):
    """Gemma3n decoder with AltUp 4D processing and PLE support."""

    layers: ModuleList
    layer_norm: LayerNorm
    altup_projections: ModuleList
    altup_unembed_projections: ModuleList
    num_altup_inputs: int
    model_dim: int
    _has_kv_projection_sharing: bool

    def __init__(
        self,
        layers: Sequence[Gemma3nDecoderLayer],
        layer_norm: LayerNorm,
        *,
        model_dim: int,
        num_altup_inputs: int = 4,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param layers: Gemma3n decoder layers.
        :param layer_norm: Final layer normalization.
        :param model_dim: Model dimensionality.
        :param num_altup_inputs: Number of AltUp versions (default 4).
        """
        super().__init__()

        self.layers = ModuleList(layers)
        self.layer_norm = layer_norm
        self.model_dim = model_dim
        self.num_altup_inputs = num_altup_inputs

        # Store per-layer KV sharing metadata so the decoder can manage slots
        # outside of FSDP-wrapped layer forwards (dict mutation side effects
        # are not guaranteed through FSDP wrappers).
        self._layer_kv_roles = [layer.kv_projection_role for layer in layers]
        self._layer_is_global = [layer.is_global for layer in layers]

        # Check if any layer uses KV projection sharing
        self._has_kv_projection_sharing = any(
            role != KVProjectionRole.NONE for role in self._layer_kv_roles
        )

        # AltUp projections: Create versions 1, 2, 3 from version 0
        self.altup_projections = ModuleList(
            [
                Linear(model_dim, model_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_altup_inputs - 1)
            ]
        )

        # AltUp unembed projections: Reverse the projections
        self.altup_unembed_projections = ModuleList(
            [
                Linear(model_dim, model_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_altup_inputs - 1)
            ]
        )

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
        per_layer_embeds: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs: Input embeddings. *Shape:* :math:`(N,S,M)`.
        :param seqs_layout: Layout information.
        :param state_bag: Incremental decoding state.
        :param per_layer_embeds: Per-layer embeddings. *Shape:* :math:`(N,S,L,256)`.
        :returns: Decoder output. *Shape:* :math:`(N,S,M)`.
        """
        kv_slots: dict[KVProjectionType, tuple[Tensor, Tensor] | None] | None = None
        if self._has_kv_projection_sharing:
            kv_slots = {
                KVProjectionType.LOCAL: None,
                KVProjectionType.GLOBAL: None,
            }

        hidden_states = self._stack_altup(seqs)

        attn_bias_cache = AttentionBiasCache()

        for layer_idx, layer in enumerate(self.layers):
            layer_ple = (
                per_layer_embeds[:, :, layer_idx, :]
                if per_layer_embeds is not None
                else None
            )

            # Resolve KV sharing args from decoder-side metadata
            pre_computed_kv = None
            kv_storage_callback = None

            if kv_slots is not None:
                role = self._layer_kv_roles[layer_idx]
                slot_key = (
                    KVProjectionType.GLOBAL
                    if self._layer_is_global[layer_idx]
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

            hidden_states = layer(
                hidden_states,
                seqs_layout,
                attn_bias_cache,
                per_layer_input=layer_ple,
                state_bag=state_bag,
                pre_computed_kv=pre_computed_kv,
                kv_storage_callback=kv_storage_callback,
            )

        seqs = self._unstack_altup(hidden_states)

        seqs = self.layer_norm(seqs)

        return seqs

    def _stack_altup(self, seqs: Tensor) -> Tensor:
        """Stack embeddings to 4D for AltUp processing.

        :param seqs: Input embeddings [B, S, M].
        :returns: 4D tensor [num_altup_inputs, B, S, M].
        """
        # Compute target magnitude for normalization
        target_magnitude = torch.mean(seqs**2, dim=-1, keepdim=True) ** 0.5
        epsilon = torch.tensor(1e-5, device=seqs.device, dtype=seqs.dtype)

        # Version 0 = original embeddings
        versions = [seqs]

        # Versions 1-3 = learned projections, magnitude-normalized
        for proj in self.altup_projections:
            projected = proj(seqs)

            # Normalize to match original magnitude
            current_magnitude = torch.mean(projected**2, dim=-1, keepdim=True)
            current_magnitude = torch.sqrt(torch.maximum(current_magnitude, epsilon))
            projected = projected * target_magnitude / current_magnitude

            versions.append(projected)

        # Stack: [4, B, S, M]
        return torch.stack(versions, dim=0)

    def _unstack_altup(self, hidden_states: Tensor) -> Tensor:
        """Unstack 4D tensor to 3D by averaging all versions.

        :param hidden_states: 4D tensor [num_altup_inputs, B, S, M].
        :returns: 3D tensor [B, S, M].
        """
        # Compute target magnitude from version 0
        target_magnitude = (
            torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        )
        epsilon = torch.tensor(
            1e-5, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # Version 0 stays as-is
        versions = [hidden_states[0]]

        # Versions 1-3: reverse projection and normalize
        for i, proj in enumerate(self.altup_unembed_projections, start=1):
            unprojected = proj(hidden_states[i])

            # Normalize to match version 0 magnitude
            current_magnitude = torch.mean(unprojected**2, dim=-1, keepdim=True)
            current_magnitude = torch.sqrt(torch.maximum(current_magnitude, epsilon))
            unprojected = unprojected * target_magnitude / current_magnitude

            versions.append(unprojected)

        # Average all versions
        stacked = torch.stack(versions, dim=0)
        return torch.mean(stacked, dim=0)

    @override
    def compile_layerwise(self, *args: Any, **kwargs: Any) -> None:
        """Compile each layer individually."""
        for layer in self.layers:
            layer.compile(*args, **kwargs)

        if self.layer_norm is not None:
            self.layer_norm.compile(*args, **kwargs)

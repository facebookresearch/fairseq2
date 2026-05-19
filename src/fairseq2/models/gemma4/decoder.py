# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 decoder stack with KV-sharing orchestration.

Iterates :class:`Gemma4DecoderLayer` instances and manages the
``LOCAL``/``GLOBAL`` KV slots so that ``SOURCE`` layers expose their key/value
tensors to downstream ``CONSUMER`` layers via a callback mechanism.  Role
assignment is performed by
:func:`fairseq2.models.gemma4.config.get_kv_projection_role`.

Unlike :class:`~fairseq2.models.gemma3n.decoder.Gemma3nDecoder`, this decoder
has **no** AltUp projections.  Input and output are plain 3-D tensors
``(B, S, M)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from torch import Tensor
from torch.nn import Module, ModuleList

from fairseq2.models.gemma3n.kv_projection import (
    KVProjectionRole,
    KVProjectionType,
)
from fairseq2.models.gemma4.decoder_layer import Gemma4DecoderLayer
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm

__all__ = ["Gemma4Decoder"]


class Gemma4Decoder(Module):
    """Gemma 4 decoder stack with KV-sharing support."""

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

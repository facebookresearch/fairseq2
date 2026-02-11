# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Key-Value sharing mechanism for Gemma3n decoder layers.

Gemma3n uses KV sharing where later layers (15-29) reuse key and value tensors
from earlier layers instead of computing their own. This saves both parameters
(no k_proj/v_proj weights) and computation during forward passes.
"""

from __future__ import annotations

from typing import final

from torch import Tensor


@final
class KVSharedLayerRegistry:
    """Registry for sharing K/V tensors between decoder layers in a forward pass.

    This implements Gemma3n's KV sharing mechanism where layers 15-29 reuse
    key and value tensors from specific earlier layers (0-14) instead of
    computing their own projections.

    NOTE: This is NOT the same as incremental decoding cache. KV sharing happens
    within a single forward pass, while incremental cache is across multiple
    generation steps.

    Example:
        >>> registry = KVSharedLayerRegistry()
        >>> # Source layer stores its K/V
        >>> registry.store_kv_for_sharing(13, key_states, value_states)
        >>> # Shared layer retrieves K/V
        >>> k, v = registry.retrieve_shared_kv(consumer_layer=20, source_layer=13)
    """

    _shared_kv: dict[int, tuple[Tensor, Tensor]]
    _source_layers_computed: set[int]

    def __init__(self) -> None:
        """Initialize an empty KV sharing registry."""
        self._shared_kv = {}
        self._source_layers_computed = set()

    def store_kv_for_sharing(
        self,
        source_layer_idx: int,
        key_states: Tensor,
        value_states: Tensor,
    ) -> None:
        """Store K/V tensors from a source layer for shared layers to retrieve.

        Source layers (typically layers 0-14) call this to make their K/V
        tensors available to shared layers (15-29).

        :param source_layer_idx:
            The index of the layer storing its K/V. Should be < first_shared_layer.
        :param key_states:
            Key tensor of shape [batch, num_heads, seq_len, head_dim].
            If using incremental decoding, this should include cached keys.
        :param value_states:
            Value tensor of shape [batch, num_heads, seq_len, head_dim].
            If using incremental decoding, this should include cached values.

        :raises RuntimeError:
            If this source layer has already stored K/V in this forward pass.
            Each source layer should only call this once per forward.
        """
        if source_layer_idx in self._shared_kv:
            raise RuntimeError(
                f"Layer {source_layer_idx} attempted to store K/V for sharing, "
                f"but it has already stored K/V in this forward pass. "
                f"Each source layer should only call store_kv_for_sharing() once."
            )

        self._shared_kv[source_layer_idx] = (key_states, value_states)
        self._source_layers_computed.add(source_layer_idx)

    def retrieve_shared_kv(
        self,
        consumer_layer_idx: int,
        source_layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve K/V tensors from a source layer.

        Shared layers (typically layers 15-29) call this to get K/V tensors
        from their designated source layer instead of computing their own.

        :param consumer_layer_idx:
            The index of the layer requesting shared K/V. Used for error messages.
        :param source_layer_idx:
            The index of the source layer whose K/V to retrieve.

        :returns:
            A tuple of (key_states, value_states) tensors.

        :raises RuntimeError:
            If the source layer hasn't stored K/V yet. This indicates either:
            - Incorrect execution order (consumer before source)
            - Misconfigured KV sharing (wrong source layer index)
        """
        if source_layer_idx not in self._source_layers_computed:
            raise RuntimeError(
                f"Layer {consumer_layer_idx} tried to retrieve shared K/V from "
                f"layer {source_layer_idx}, but that source layer has not stored "
                f"K/V yet. This indicates either incorrect execution order "
                f"(layers must execute sequentially) or misconfigured KV sharing "
                f"(consumer layer {consumer_layer_idx} is configured to share from "
                f"non-existent or later layer {source_layer_idx})."
            )

        key_states, value_states = self._shared_kv[source_layer_idx]
        return key_states, value_states

    def clear(self) -> None:
        """Clear the registry for the next forward pass.

        Should be called at the end of each forward pass to reset state.
        The decoder typically calls this automatically.
        """
        self._shared_kv.clear()
        self._source_layers_computed.clear()

    def __repr__(self) -> str:
        """Return string representation showing stored source layers."""
        return (
            f"KVSharedLayerRegistry("
            f"source_layers={sorted(self._source_layers_computed)})"
        )

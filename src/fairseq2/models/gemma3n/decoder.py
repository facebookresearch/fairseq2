# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, final

import torch
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma3n.decoder_layer import Gemma3nDecoderLayer
from fairseq2.models.gemma3n.kv_projection import KVProjectionRole, KVProjectionType
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.models.transformer_lm import TransformerLMDecoder
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm
from fairseq2.nn.projection import Linear


@final
class Gemma3nDecoder(TransformerLMDecoder):
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

        # Check if any layer uses KV projection sharing
        self._has_kv_projection_sharing = any(
            layer.kv_projection_role != KVProjectionRole.NONE for layer in layers
        )

        # AltUp projections: Create versions 1, 2, 3 from version 0
        self.altup_projections = ModuleList([
            Linear(model_dim, model_dim, bias=False, device=device, dtype=dtype)
            for _ in range(num_altup_inputs - 1)
        ])

        # AltUp unembed projections: Reverse the projections
        self.altup_unembed_projections = ModuleList([
            Linear(model_dim, model_dim, bias=False, device=device, dtype=dtype)
            for _ in range(num_altup_inputs - 1)
        ])

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs: Input sequences [B, S, M] from frontend.
        :param seqs_layout: Batch layout information.
        :param state_bag: State bag containing PLE embeddings and incremental state.
        :returns: Output sequences [B, S, M].
        """
        # Create KV projection slots if sharing is enabled
        kv_projection_slots = None
        if self._has_kv_projection_sharing:
            kv_projection_slots = {
                KVProjectionType.LOCAL: None,
                KVProjectionType.GLOBAL: None,
            }

        # Stack to 4D: [B, S, M] → [4, B, S, M]
        hidden_states = self._stack_altup(seqs)

        # Get PLE embeddings from state_bag (set by frontend)
        per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

        attn_bias_cache = AttentionBiasCache()

        for layer_idx, layer in enumerate(self.layers):
            # Extract PLE for this layer
            if per_layer_inputs is not None:
                layer_ple = per_layer_inputs[:, :, layer_idx, :]
            else:
                layer_ple = None

            hidden_states = layer(
                hidden_states,
                seqs_layout,
                attn_bias_cache,
                per_layer_input=layer_ple,
                state_bag=state_bag,
                kv_projection_slots=kv_projection_slots,
            )

        # Unstack to 3D: [4, B, S, M] → [B, S, M]
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
        target_magnitude = torch.mean(hidden_states[0]**2, dim=-1, keepdim=True) ** 0.5
        epsilon = torch.tensor(1e-5, device=hidden_states.device, dtype=hidden_states.dtype)

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

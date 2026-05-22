# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 decoder layer.

Implements :class:`Gemma4DecoderLayer` following the HuggingFace
``Gemma4TextDecoderLayer`` forward logic (``modeling_gemma4.py`` lines 1381-1438)
using fairseq2 abstractions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models.transformer import (
    AttentionBiasCache,
    FeedForwardNetwork,
    MultiheadAttention,
)
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm
from fairseq2.nn.projection import Linear


class Gemma4DecoderLayer(Module):
    """Gemma 4 decoder layer with optional MoE and Per-Layer Embeddings (PLE).

    The layer follows a pre-norm architecture with five sequential stages:

    1. **Self-attention** -- input layer-norm, attention, post-attention norm,
       then additive residual.
    2. **Dense FFN** -- pre-FFN norm, MLP.
    3. **Optional MoE** -- when present, runs in parallel to the dense FFN.
       The router selects top-k experts, and the sparse expert output is
       combined with the dense MLP output via additional norms.
    4. **Optional PLE** -- Per-Layer Embedding gating, projection, and norm
       with an additive residual.
    5. **Layer scalar** -- element-wise scaling of the output.
    """

    # Core modules
    self_attn: MultiheadAttention
    ffn: FeedForwardNetwork
    input_layernorm: LayerNorm
    post_attention_layernorm: LayerNorm
    pre_feedforward_layernorm: LayerNorm
    post_feedforward_layernorm: LayerNorm

    # Optional PLE modules
    per_layer_input_gate: Linear | None
    per_layer_projection: Linear | None
    post_per_layer_input_norm: LayerNorm | None

    # Optional MoE modules
    router: Module | None
    experts: Module | None
    post_feedforward_layernorm_1: LayerNorm | None
    pre_feedforward_layernorm_2: LayerNorm | None
    post_feedforward_layernorm_2: LayerNorm | None

    # Flags
    enable_moe: Final[bool]
    enable_ple: Final[bool]

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
        pre_feedforward_layernorm: LayerNorm,
        post_feedforward_layernorm: LayerNorm,
        # Optional PLE components
        per_layer_input_gate: Linear | None = None,
        per_layer_projection: Linear | None = None,
        post_per_layer_input_norm: LayerNorm | None = None,
        # Optional MoE components
        router: Module | None = None,
        experts: Module | None = None,
        post_feedforward_layernorm_1: LayerNorm | None = None,
        pre_feedforward_layernorm_2: LayerNorm | None = None,
        post_feedforward_layernorm_2: LayerNorm | None = None,
        # Layer scalar
        layer_scalar_init: float = 1.0,
        # Activation for PLE
        activation_fn: str = "gelu_pytorch_tanh",
    ) -> None:
        """
        :param self_attn: The multi-head self-attention module.
        :param ffn: The dense feed-forward network.
        :param input_layernorm: Pre-attention layer normalization.
        :param post_attention_layernorm: Post-attention layer normalization.
        :param pre_feedforward_layernorm: Pre-FFN layer normalization.
        :param post_feedforward_layernorm: Post-FFN layer normalization (applied
            after the dense-MLP path, or after dense+MoE merge).
        :param per_layer_input_gate: PLE gating projection (model_dim -> model_dim).
        :param per_layer_projection: PLE output projection (ple_dim -> model_dim).
        :param post_per_layer_input_norm: PLE post-normalization.
        :param router: MoE router module. Expected to return
            ``(logits, top_k_weights, top_k_indices)`` when called with
            ``(T, D)`` input.
        :param experts: MoE experts module. Expected to accept
            ``(hidden_states, top_k_indices, top_k_weights)`` and return
            ``(T, D)`` output.
        :param post_feedforward_layernorm_1: Norm applied to dense MLP output
            before merging with MoE output.
        :param pre_feedforward_layernorm_2: Norm applied to flattened residual
            before feeding into MoE experts.
        :param post_feedforward_layernorm_2: Norm applied to MoE expert output
            before merging with dense MLP output.
        :param layer_scalar_init: Initial value for the learned layer scalar.
        :param activation_fn: Activation function name for PLE gating. Only
            ``"gelu_pytorch_tanh"`` is currently supported.
        """
        super().__init__()

        # Core modules
        self.self_attn = self_attn
        self.ffn = ffn

        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

        # PLE modules
        self.enable_ple = per_layer_input_gate is not None
        if self.enable_ple:
            if per_layer_projection is None:
                raise ValueError(
                    "`per_layer_projection` must be provided when "
                    "`per_layer_input_gate` is not None."
                )
            if post_per_layer_input_norm is None:
                raise ValueError(
                    "`post_per_layer_input_norm` must be provided when "
                    "`per_layer_input_gate` is not None."
                )
            self.per_layer_input_gate = per_layer_input_gate
            self.per_layer_projection = per_layer_projection
            self.post_per_layer_input_norm = post_per_layer_input_norm
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # MoE modules
        self.enable_moe = router is not None
        if self.enable_moe:
            if experts is None:
                raise ValueError(
                    "`experts` must be provided when `router` is not None."
                )
            if post_feedforward_layernorm_1 is None:
                raise ValueError(
                    "`post_feedforward_layernorm_1` must be provided when "
                    "`router` is not None."
                )
            if pre_feedforward_layernorm_2 is None:
                raise ValueError(
                    "`pre_feedforward_layernorm_2` must be provided when "
                    "`router` is not None."
                )
            if post_feedforward_layernorm_2 is None:
                raise ValueError(
                    "`post_feedforward_layernorm_2` must be provided when "
                    "`router` is not None."
                )
            self.router = router
            self.experts = experts
            self.post_feedforward_layernorm_1 = post_feedforward_layernorm_1
            self.pre_feedforward_layernorm_2 = pre_feedforward_layernorm_2
            self.post_feedforward_layernorm_2 = post_feedforward_layernorm_2
        else:
            self.router = None
            self.experts = None
            self.post_feedforward_layernorm_1 = None
            self.pre_feedforward_layernorm_2 = None
            self.post_feedforward_layernorm_2 = None

        # Layer scalar (non-trainable buffer)
        self.register_buffer("layer_scalar", torch.ones(1) * layer_scalar_init)

        # Activation function for PLE
        self._activation_fn = activation_fn

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        bias_cache: AttentionBiasCache,
        per_layer_input: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        pre_computed_kv: tuple[Tensor, Tensor] | None = None,
        kv_storage_callback: Callable[[Tensor, Tensor], None] | None = None,
    ) -> Tensor:
        """Run one decoder layer.

        :param seqs: Hidden states. *Shape:* :math:`(N, S, D)` where :math:`N`
            is the batch size, :math:`S` the sequence length, and :math:`D` the
            model dimensionality.
        :param seqs_layout: Batch layout for attention masking.
        :param bias_cache: Attention bias cache (causal mask, etc.).
        :param per_layer_input: Per-layer embedding input for PLE.
            *Shape:* :math:`(N, S, D_{ple})`. Required when PLE is enabled.
        :param state_bag: Incremental state bag for KV-cache during generation.
        :param pre_computed_kv: Pre-computed ``(K, V)`` tensors from a SOURCE
            layer for KV sharing.  Passed through to the attention module.
        :param kv_storage_callback: Callback invoked with ``(K, V)`` after
            attention computation so that a SOURCE layer can store them for
            downstream CONSUMERs.
        :returns: Decoder layer output. *Shape:* same as ``seqs``.
        """
        hidden_states = seqs

        # ---- 1. Self-attention ----
        hidden_states = self._forward_self_attn(
            hidden_states,
            seqs_layout,
            bias_cache,
            state_bag,
            pre_computed_kv=pre_computed_kv,
            kv_storage_callback=kv_storage_callback,
        )

        # ---- 2 & 3. FFN (+ optional MoE) ----
        hidden_states = self._forward_ffn(hidden_states)

        # ---- 4. Optional PLE ----
        if self.enable_ple and per_layer_input is not None:
            hidden_states = self._forward_ple(hidden_states, per_layer_input)

        # ---- 5. Layer scalar ----
        hidden_states = hidden_states * self.layer_scalar  # type: ignore[operator]

        return hidden_states

    def _forward_self_attn(
        self,
        hidden_states: Tensor,
        seqs_layout: BatchLayout,
        bias_cache: AttentionBiasCache,
        state_bag: IncrementalStateBag | None,
        *,
        pre_computed_kv: tuple[Tensor, Tensor] | None = None,
        kv_storage_callback: Callable[[Tensor, Tensor], None] | None = None,
    ) -> Tensor:
        """Self-attention with pre-norm and post-attention norm."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            seqs=hidden_states,
            seqs_layout=seqs_layout,
            keys=hidden_states,
            keys_layout=seqs_layout,
            values=hidden_states,
            bias_cache=bias_cache,
            state_bag=state_bag,
            pre_computed_kv=pre_computed_kv,  # type: ignore[call-arg]
            kv_storage_callback=kv_storage_callback,  # type: ignore[call-arg]
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_ffn(self, hidden_states: Tensor) -> Tensor:
        """Dense FFN with optional parallel MoE branch."""
        residual = hidden_states

        # Dense MLP
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)

        # Optional MoE (parallel to dense MLP)
        if self.enable_moe:
            assert self.router is not None
            assert self.experts is not None
            assert self.post_feedforward_layernorm_1 is not None
            assert self.pre_feedforward_layernorm_2 is not None
            assert self.post_feedforward_layernorm_2 is not None

            # Norm the dense MLP output
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # Route using the PRE-MLP residual (flattened to 2D)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_indices = self.router(hidden_states_flat)

            # Norm and run experts
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(
                hidden_states_2, top_k_indices, top_k_weights
            )

            # Reshape back to 3D and norm
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Merge dense + MoE
            hidden_states = hidden_states_1 + hidden_states_2

        # Final post-FFN norm + residual
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_ple(self, hidden_states: Tensor, per_layer_input: Tensor) -> Tensor:
        """Per-Layer Embedding (PLE) augmentation."""
        assert self.per_layer_input_gate is not None
        assert self.per_layer_projection is not None
        assert self.post_per_layer_input_norm is not None

        residual = hidden_states

        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        parts = [
            f"enable_moe={self.enable_moe}",
            f"enable_ple={self.enable_ple}",
        ]
        return ", ".join(parts)

    if TYPE_CHECKING:
        __call__ = forward

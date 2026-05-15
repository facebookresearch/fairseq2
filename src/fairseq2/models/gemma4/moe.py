# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture-of-Experts modules for Gemma 4.

Gemma 4 MoE is *additive parallel*: the MoE output is summed with the dense MLP
output in the decoder layer rather than replacing it.  The integration happens
outside this module.

Classes:
    - :class:`Gemma4Router`  -- RMSNorm + scaled projection + per-expert scaling
    - :class:`Gemma4Experts` -- fused 3-D parameter experts with GELU activation
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.nn import Linear, RMSNorm


class Gemma4Router(Module):
    """Top-k router for Gemma 4 MoE with RMSNorm and per-expert scaling.

    The routing pipeline is:

    1.  RMSNorm the input (no learnable affine -- ``elementwise_affine=False``).
    2.  Element-wise multiply by a learnable ``scale`` vector and a constant
        ``scalar_root_size = model_dim ** -0.5``.
    3.  Project to ``num_experts`` logits via a bias-free linear layer.
    4.  Softmax over experts, then select top-k.
    5.  Renormalise selected weights to sum to 1, then multiply by
        ``per_expert_scale``.

    Reference: ``Gemma4TextRouter`` in HuggingFace ``modeling_gemma4.py``.
    """

    model_dim: Final[int]
    num_experts: Final[int]
    top_k: Final[int]
    scalar_root_size: Final[float]

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        top_k: int,
        *,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (``hidden_size``).
        :param num_experts:
            The total number of routed experts.
        :param top_k:
            The number of experts activated per token.
        :param rms_norm_eps:
            Epsilon for the RMSNorm layer.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.scalar_root_size = model_dim ** -0.5

        # RMSNorm without learnable weight (with_scale=False in HF).
        self.norm = RMSNorm(model_dim, bias=False, eps=rms_norm_eps, elementwise_affine=False)

        # Linear projection to expert logits (no bias).
        self.proj = Linear(model_dim, num_experts, bias=False)

        # Learnable per-dimension scale applied after normalisation.
        self.scale = Parameter(torch.ones(model_dim))

        # Learnable per-expert scale applied to the final routing weights.
        self.per_expert_scale = Parameter(torch.ones(num_experts))

    def forward(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param hidden_states:
            Token representations of shape ``(T, D)`` where *T* is the
            (flattened) number of tokens.

        :returns:
            A 3-tuple of:

            - ``router_probs``   -- full softmax probabilities  ``(T, E)``
            - ``top_k_weights``  -- scaled top-k weights        ``(T, K)``
            - ``top_k_indices``  -- selected expert indices     ``(T, K)``
        """
        # 1. Normalise and scale.
        x = self.norm(hidden_states)
        x = x * self.scale * self.scalar_root_size

        # 2. Project to expert scores and apply softmax.
        expert_scores = self.proj(x)  # (T, E)
        router_probs = F.softmax(expert_scores, dim=-1)

        # 3. Select top-k experts.
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )

        # 4. Renormalise so the selected weights sum to 1.
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 5. Apply per-expert scale.
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

        return router_probs, top_k_weights, top_k_indices


class Gemma4Experts(Module):
    """Fused expert layer with 3-D weight parameters for Gemma 4 MoE.

    Each expert is a gated MLP (gate + up -> activation -> down) stored as a
    single ``(E, 2*I, D)`` gate-up projection and a ``(E, D, I)`` down
    projection.  Unlike Qwen/LLaMA MoE, Gemma 4 uses GELU (with tanh
    approximation) instead of SiLU.

    Reference: ``Gemma4TextExperts`` in HuggingFace ``modeling_gemma4.py``.
    """

    num_experts: Final[int]
    model_dim: Final[int]
    moe_intermediate_size: Final[int]

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        moe_intermediate_size: int,
        *,
        activation_fn: str = "gelu_pytorch_tanh",
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (``hidden_size``).
        :param num_experts:
            The total number of routed experts.
        :param moe_intermediate_size:
            The intermediate (inner) dimension of each expert's FFN.
        :param activation_fn:
            The activation function name.  ``"gelu_pytorch_tanh"`` maps to
            ``torch.nn.functional.gelu(..., approximate="tanh")``.
        """
        super().__init__()

        self.num_experts = num_experts
        self.model_dim = model_dim
        self.moe_intermediate_size = moe_intermediate_size

        # Resolve activation.
        if activation_fn == "gelu_pytorch_tanh":
            self._act_fn = self._gelu_tanh
        else:
            raise ValueError(
                f"Unsupported activation_fn: '{activation_fn}'.  "
                "Expected 'gelu_pytorch_tanh'."
            )

        # Fused gate + up projection: (E, 2*I, D)
        self.gate_up_proj = Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, model_dim)
        )

        # Down projection: (E, D, I)
        self.down_proj = Parameter(
            torch.empty(num_experts, model_dim, moe_intermediate_size)
        )

    @staticmethod
    def _gelu_tanh(x: Tensor) -> Tensor:
        return F.gelu(x, approximate="tanh")

    def forward(
        self,
        hidden_states: Tensor,
        top_k_indices: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """
        :param hidden_states:
            Token representations of shape ``(T, D)``.
        :param top_k_indices:
            Selected expert indices of shape ``(T, K)``.
        :param top_k_weights:
            Routing weights of shape ``(T, K)``.

        :returns:
            Expert-mixed output of shape ``(T, D)``.
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        # Build per-expert token masks.
        with torch.no_grad():
            # (T, K, E)
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
            # (E, K, T)
            expert_mask = expert_mask.permute(2, 1, 0)
            # Identify which experts received at least one token.
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            # Gather tokens assigned to this expert.
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[token_idx]

            # Gate + up projection, then split.
            gate, up = F.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)

            # Activation (GELU tanh) and gating.
            current_hidden_states = self._act_fn(gate) * up

            # Down projection.
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )

            # Weight by routing score.
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )

            # Scatter-add back into the output.
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states

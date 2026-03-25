# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture-of-Experts modules for Qwen 3.5 MoE.

This module implements the MoE architecture from Qwen 3.5 MoE following the
HuggingFace reference in ``modeling_qwen3_5_moe.py``.

Classes:
    - :class:`Qwen35TopKRouter`  — softmax → top-k → renormalize  (HF lines 841-857)
    - :class:`Qwen35Experts`     — fused 3-D parameter experts     (HF lines 802-838)
    - :class:`Qwen35MoeBlock`    — router + experts + shared expert (HF lines 860-879)
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.models.transformer import FeedForwardNetwork, GLUFeedForwardNetwork
from fairseq2.nn import Linear


class Qwen35TopKRouter(Module):
    """Top-k softmax router for Qwen 3.5 MoE.

    Computes softmax over all experts, selects the top-k, and renormalises the
    selected weights so they sum to 1.

    Reference: ``Qwen3_5MoeTopKRouter`` (HF lines 841-857).
    """

    num_experts: Final[int]
    top_k: Final[int]
    model_dim: Final[int]

    def __init__(self, num_experts: int, top_k: int, model_dim: int) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.model_dim = model_dim

        self.weight = Parameter(torch.zeros(num_experts, model_dim))

    def forward(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param hidden_states:
            Token representations of shape ``(T, D)`` where *T* is the
            (flattened) number of tokens.

        :returns:
            A 3-tuple of:
            - ``router_logits``  — full softmax distribution ``(T, E)``
            - ``router_weights`` — renormalised top-k weights  ``(T, K)``
            - ``router_indices`` — selected expert indices     ``(T, K)``
        """
        hidden_states = hidden_states.reshape(-1, self.model_dim)

        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        router_weights, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        router_weights = router_weights / router_weights.sum(
            dim=-1, keepdim=True
        )
        router_weights = router_weights.to(router_logits.dtype)

        return router_logits, router_weights, router_indices


class Qwen35Experts(Module):
    """Fused expert layer with 3-D weight parameters for Qwen 3.5 MoE.

    Each expert is a GLU-style MLP (gate+up → SiLU → down) stored as a single
    ``(E, 2*I, D)`` gate-up projection and a ``(E, D, I)`` down projection so
    that individual experts can be indexed without slicing overhead.

    Reference: ``Qwen3_5MoeExperts`` (HF lines 802-838).
    """

    num_experts: Final[int]
    model_dim: Final[int]
    expert_inner_dim: Final[int]

    def __init__(
        self,
        num_experts: int,
        model_dim: int,
        expert_inner_dim: int,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.model_dim = model_dim
        self.expert_inner_dim = expert_inner_dim

        self.gate_up_proj = Parameter(
            torch.empty(num_experts, 2 * expert_inner_dim, model_dim)
        )
        self.down_proj = Parameter(
            torch.empty(num_experts, model_dim, expert_inner_dim)
        )

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
            Renormalised routing weights of shape ``(T, K)``.

        :returns:
            Expert-mixed output of shape ``(T, D)``.
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
            # (T, K, E) → (E, K, T)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[token_idx]

            gate, up = F.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)

            current_hidden_states = F.silu(gate) * up

            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )

            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states


class Qwen35MoeBlock(FeedForwardNetwork):
    """Sparse Mixture-of-Experts feed-forward block for Qwen 3.5 MoE.

    Combines a top-k router, a set of sparse experts, a shared expert (standard
    GLU MLP), and a learned sigmoid gate that blends the shared expert output
    into the final result.

    This class inherits from :class:`FeedForwardNetwork` so it can serve as a
    drop-in replacement for :class:`GLUFeedForwardNetwork` inside any
    Transformer decoder layer.

    Reference: ``Qwen3_5MoeSparseMoeBlock`` (HF lines 860-879).
    """

    model_dim: Final[int]

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (``hidden_size``).
        :param num_experts:
            The total number of routed experts.
        :param num_experts_per_tok:
            The number of experts activated per token (top-k).
        :param moe_intermediate_size:
            The intermediate (inner) dimension of each routed expert.
        :param shared_expert_intermediate_size:
            The intermediate (inner) dimension of the shared expert.
        """
        super().__init__()

        self.model_dim = model_dim

        self.gate = Qwen35TopKRouter(num_experts, num_experts_per_tok, model_dim)

        self.experts = Qwen35Experts(
            num_experts, model_dim, moe_intermediate_size
        )

        self.shared_expert = GLUFeedForwardNetwork(
            model_dim,
            shared_expert_intermediate_size,
            bias=False,
            inner_dim_scale=1.0,
        )

        self.shared_expert_gate = Linear(model_dim, 1, bias=False)

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        B, S, D = seqs.shape

        hidden_states = seqs.view(-1, D)

        shared_out = self.shared_expert(hidden_states)

        _, routing_weights, selected_experts = self.gate(hidden_states)

        expert_out = self.experts(
            hidden_states, selected_experts, routing_weights
        )

        shared_out = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        )

        return (expert_out + shared_out).reshape(B, S, D)

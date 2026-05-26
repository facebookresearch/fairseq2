# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Experts (MoE) module for NemotronH.

This module implements the NemotronH-style MoE with:
- Sigmoid routing (NOT softmax) with bias correction
- Group-level top-K pre-selection
- Squared ReLU activation in expert MLPs
- Shared expert always active
- Routing weight normalization and scaling by routed_scaling_factor

Key differences from Llama4/Qwen MoE:
- Uses sigmoid instead of softmax for routing
- Has e_score_correction_bias buffer
- Uses squared ReLU (relu(x)^2) instead of SiLU/SwiGLU
- Separate nn.ModuleList of expert instances (not fused 3D params)
"""

from __future__ import annotations

from typing import final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from fairseq2.gang import Gang


class SquaredReLU(nn.Module):
    """Squared ReLU activation: relu(x)^2."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x).square()


@final
class NemotronHExpert(nn.Module):
    """Single expert MLP: up_proj -> squared_relu -> down_proj.

    Unlike SwiGLU/GLU experts, this uses a simple two-layer MLP
    with squared ReLU activation (no gate projection).
    """

    def __init__(
        self,
        model_dim: int,
        intermediate_size: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(model_dim, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, model_dim, bias=bias)
        self.act = SquaredReLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


@final
class NemotronHTopKRouter(nn.Module):
    """Sigmoid router with bias correction for NemotronH MoE.

    Routing flow:
    1. Compute sigmoid scores: sigmoid(linear(hidden_states))
    2. Add e_score_correction_bias for routing decisions
    3. Group-level pre-selection (if n_group > 1)
    4. Token-level top-K selection
    5. Gather original sigmoid scores (without bias)
    6. Normalize to sum=1, then scale by routed_scaling_factor
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        *,
        top_k: int = 6,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        # Router projection
        self.weight = nn.Parameter(torch.empty(num_experts, model_dim))

        # Bias correction buffer (fp32)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)

    @override
    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tensor. Shape: [batch*seq_len, model_dim]

        Returns:
            Tuple of:
            - routing_weights: Normalized weights for selected experts.
                Shape: [batch*seq_len, top_k]
            - selected_experts: Indices of selected experts.
                Shape: [batch*seq_len, top_k]
        """
        # Compute sigmoid routing scores (in float32 for stability)
        router_logits = torch.sigmoid(
            F.linear(hidden_states.float(), self.weight.float())
        )  # [num_tokens, num_experts]

        # Add bias correction for routing decisions
        scores_for_choice = router_logits + self.e_score_correction_bias

        # Group-level pre-selection
        if self.n_group > 1 and self.topk_group > 1:
            # Reshape to groups
            group_scores = scores_for_choice.view(
                -1, self.n_group, self.num_experts // self.n_group
            )
            # Get top scores per group
            group_max = group_scores.amax(dim=-1)  # [num_tokens, n_group]
            # Select top groups
            _, top_groups = group_max.topk(self.topk_group, dim=-1)
            # Create mask for selected groups
            group_mask = torch.zeros_like(group_max, dtype=torch.bool)
            group_mask.scatter_(1, top_groups, True)
            # Mask out non-selected groups
            group_mask = group_mask.unsqueeze(-1).expand_as(group_scores)
            scores_for_choice = scores_for_choice.view_as(group_scores)
            scores_for_choice = scores_for_choice.masked_fill(~group_mask, float("-inf"))
            scores_for_choice = scores_for_choice.view(-1, self.num_experts)

        # Token-level top-K selection
        _, selected_experts = scores_for_choice.topk(self.top_k, dim=-1)

        # Gather original sigmoid scores (without bias) for the selected experts
        routing_weights = router_logits.gather(1, selected_experts)

        # Normalize
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Scale
        routing_weights = routing_weights * self.routed_scaling_factor

        return routing_weights, selected_experts


@final
class NemotronHMoE(nn.Module):
    """Mixture of Experts block for NemotronH.

    Consists of:
    - A sigmoid top-K router with bias correction
    - 128 routed experts (each: up_proj -> squared_relu -> down_proj)
    - 1 shared expert (always active, larger intermediate size)

    Output = sum(routing_weight_i * expert_i(x)) + shared_expert(x)
    """

    def __init__(
        self,
        model_dim: int,
        *,
        num_experts: int = 128,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 1856,
        shared_expert_intermediate_size: int = 3712,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router
        self.gate = NemotronHTopKRouter(
            model_dim,
            num_experts,
            top_k=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [
                NemotronHExpert(model_dim, moe_intermediate_size, bias=bias)
                for _ in range(num_experts)
            ]
        )

        # Shared expert (always active)
        self.shared_experts = NemotronHExpert(
            model_dim, shared_expert_intermediate_size, bias=bias
        )

        # TODO: Implement all-reduce when tensor parallel sharding is added.
        self.tp_gang: Gang | None = None

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of MoE block.

        Args:
            hidden_states: Input tensor. Shape: [batch, seq_len, model_dim]

        Returns:
            Output tensor. Shape: [batch, seq_len, model_dim]
        """
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]

        # Flatten to [num_tokens, model_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # Route
        routing_weights, selected_experts = self.gate(hidden_states_flat)
        # routing_weights: [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]

        # Compute shared expert output
        shared_output = self.shared_experts(hidden_states_flat)

        # Compute routed expert outputs
        # Use token-level dispatch (efficient for moderate number of active experts)
        # Accumulate in float32 for numerical stability (HF does the same)
        final_output = torch.zeros(
            num_tokens, hidden_dim,
            dtype=torch.float32,
            device=hidden_states_flat.device,
        )

        # Precompute which experts have tokens (avoids looping over all 128)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
            # expert_mask: [num_tokens, top_k, num_experts]
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().reshape(-1)

        # Process only experts that have tokens routed to them
        for expert_idx in expert_hit:
            expert_idx_item = expert_idx.item()
            top_k_pos, token_indices = torch.where(expert_mask[expert_idx_item])

            if token_indices.numel() == 0:
                continue

            # Get the routing weights for these token-expert pairs
            weights = routing_weights[token_indices, top_k_pos]  # [num_selected]

            # Compute expert output
            expert_input = hidden_states_flat[token_indices]  # [num_selected, D]
            expert_output = self.experts[expert_idx_item](expert_input)  # [num_selected, D]

            # Weighted accumulation in float32
            final_output.index_add_(
                0,
                token_indices,
                (expert_output * weights.unsqueeze(-1)).float(),
            )

        # Cast back to input dtype, then add shared expert output
        final_output = final_output.to(hidden_states_flat.dtype)
        final_output = final_output + shared_output

        return final_output.view(orig_shape)

    @override
    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"num_experts_per_tok={self.num_experts_per_tok}"
        )

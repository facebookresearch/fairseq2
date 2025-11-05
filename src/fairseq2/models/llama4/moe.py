# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing_extensions import override

from fairseq2.gang import Gang
from fairseq2.models.transformer import (
    ExpertNetwork,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    GroupedExpertNetwork,
)
from fairseq2.ops.tensor_parallel import reduce


class MoE(FeedForwardNetwork):
    """This class implements a MoE layer (Mixture of Experts).
    Mixture of Experts typically consists of a set of expert networks,
    alongside with a router, which directs input tokens
    to the appropriate experts.
    See more details in https://arxiv.org/pdf/2407.06204.
    """

    router: nn.Parameter
    top_k: int
    experts: ExpertNetwork
    shared_expert: GLUFeedForwardNetwork | None
    tp_gang: Gang | None

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        use_shared_expert: bool,
        num_experts: int,
        capacity_factor: float,
        top_k: int,
        *,
        moe_activation: nn.Module | None = None,
        inner_dim_scale: float = 2 / 3,
        inner_dim_to_multiple: int = 1,
        auto_scale: bool = True,
    ) -> None:
        super().__init__()

        # same scaling of inner_dim as in dense Llama
        self.inner_dim_scale = inner_dim_scale

        if inner_dim_scale != 1.0:
            inner_dim = int(inner_dim * inner_dim_scale)

        self.inner_dim_to_multiple = inner_dim_to_multiple

        if auto_scale:
            inner_dim = int(inner_dim / (capacity_factor + int(use_shared_expert)))

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        if use_shared_expert:
            # Before scaling inner_dim, we create the shared expert
            self.shared_expert = GLUFeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=inner_dim,
                bias=False,
                gate_activation=None,  # Silu
                inner_dim_scale=1.0,  # no scaling
                inner_dim_to_multiple=1,  # no scaling
            )
        else:
            self.shared_expert = None

        self.router = nn.Parameter(torch.empty(model_dim, num_experts))
        self.top_k = top_k

        self.experts = GroupedExpertNetwork(
            num_experts,
            model_dim,
            inner_dim,
            activation=moe_activation,
        )

        # If set, is used at the end of the forward to reduce the output
        self.tp_gang = None

    @override
    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seqs (torch.Tensor): Input tensor with shape ``(*, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(*, dim)``.
        """
        x = seqs

        dim = x.shape[-1]

        # Apply token-choice routing

        # (num_tokens, num_experts)
        logits = x.reshape(-1, dim) @ self.router

        num_tokens = logits.shape[0]

        scores, token_indices = torch.topk(logits, self.top_k, dim=1)

        scores = (
            torch.full_like(logits, float("-inf"))
            .scatter_(1, token_indices, scores)
            .transpose(0, 1)
        )

        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .view(1, -1)
            .expand(scores.size(0), -1)
        )

        top_scores = torch.sigmoid(scores)

        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # (num_experts * tokens_per_expert, dim)
        routed_input = torch.gather(x.view(-1, dim), dim=0, index=token_indices)
        routed_input = routed_input * top_scores.reshape(-1, 1)
        routed_input = routed_input.reshape(self.experts.num_local_experts, -1, dim)

        # Compute experts output

        # (num_local_experts, tokens_per_expert * ep, dim)
        routed_output = self.experts(routed_input)
        # (num_experts * tokens_per_expert, dim)
        routed_output = routed_output.reshape(-1, dim)

        # Compute shared expert output
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(-1, dim)
        else:
            out = torch.zeros_like(x.view(-1, dim))

        # Combine expert outputs
        out.scatter_add_(dim=0, index=token_indices, src=routed_output)

        # Possibly reduce
        if self.tp_gang:
            out = reduce(out, self.tp_gang)

        out = out.reshape(seqs.shape)
        return out

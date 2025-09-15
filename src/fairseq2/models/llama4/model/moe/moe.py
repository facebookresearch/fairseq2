# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing_extensions import override

from fairseq2.gang import Gang
from fairseq2.models.llama4.model.moe.experts import Experts
from fairseq2.models.llama4.model.moe.router import Router
from fairseq2.models.llama4.model.moe.token_dispatcher import (
    MoETokenDispatcher,
    TPTokenDispatcher,
)
from fairseq2.models.transformer import FeedForwardNetwork, GLUFeedForwardNetwork
from fairseq2.ops.tensor_parallel import reduce


class MoE(FeedForwardNetwork):
    """This class implements the MoE layer (Mixture of Experts). Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.
    """

    router: Router
    experts: Experts
    shared_expert: GLUFeedForwardNetwork | None
    token_dispatcher: MoETokenDispatcher
    tp_gang: Gang | None

    eval_with_saved_stats: bool

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
        running_stats_ema: float = 0.99,
        eval_with_saved_stats: bool = True,
        expert_act_threshold: float = 0.0,
    ) -> None:
        super().__init__(model_dim)

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

        self.eval_with_saved_stats = eval_with_saved_stats

        self.running_stats_ema = running_stats_ema

        # running stats for each expert shape (mean, var*count, count)
        self.register_buffer(
            "running_gate_stats",
            torch.zeros(3, num_experts, dtype=torch.float32),
        )

        self.register_buffer(
            "global_gate_stats",
            torch.zeros(3, num_experts, dtype=torch.float32),
        )

        self.router = Router(
            model_dim=model_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_act_threshold=expert_act_threshold,
        )

        self.experts = Experts(
            model_dim=model_dim,
            inner_dim=inner_dim,
            num_local_experts=num_experts,
            activation=moe_activation,
        )

        # Use a TP token dispatcher for non-EP scenarios
        self.token_dispatcher = TPTokenDispatcher(num_experts)

        # If set, is used at the end of the forward to reduce the output
        self.tp_gang = None

    @override
    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seqs (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        x = seqs
        # Update running stats for experts threshold
        self.running_gate_stats = self.running_gate_stats.float()  # type: ignore[has-type]
        self.running_gate_stats.data = self.running_gate_stats.data.to(x.device)
        self.global_gate_stats = self.global_gate_stats.float()  # type: ignore[has-type]
        self.global_gate_stats.data = self.global_gate_stats.data.to(x.device)

        experts_threshold = None
        if not self.training and self.eval_with_saved_stats:
            # Assume std is 0, just take the mean of the scores as the threshold
            experts_threshold = self.global_gate_stats[0]

        bs, slen, dim = x.shape
        # top_scores shape (num_experts, tokens_per_expert) for EC and (bs*slen*experts_per_token, ) for TC
        # selected_indices shape (num_experts, tokens_per_expert) for EC training,
        # (num_experts, bs*slen) for EC inference, and (bs*slen*experts_per_token,) for TC
        (
            top_scores,
            token_indices,
            num_local_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim), experts_threshold=experts_threshold)

        tokens_per_expert = top_scores.shape[1]

        # token_indices shape (num_experts*tokens_per_expert, dim) for training and (num_experts*bs*slen, dim) for inference
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)
        # routed_input shape (num_experts*tokens_per_expert, dim)
        routed_input = torch.gather(x.view(-1, dim), dim=0, index=token_indices)
        routed_input = routed_input * top_scores.reshape(-1, 1)

        # Dispatch of tokens to experts (possibly with all-to-all)
        # permute_metadata is only useful if sharding_strategy == "dp2ep-v1"
        permute_metadata = {"nt": {"tokens_per_expert": tokens_per_expert}}
        routed_input = self.token_dispatcher.token_permutation(
            routed_input,
            permute_metadata=permute_metadata,  # type: ignore[arg-type]
        )

        # Meanwhile, we run the first step of the shared expert
        if self.shared_expert is not None:
            shared_expert_out = self.shared_expert.forward_gateinner(x)
        else:
            shared_expert_out = None

        # Wait for tokens to be ready
        # routed_input shape: (num_local_experts * tokens_per_expert, dim)
        routed_input = self.token_dispatcher.wait_token_permutation(
            routed_input,
            permute_metadata=permute_metadata,  # type: ignore[arg-type]
        )

        if num_local_tokens_per_expert is None:  # non-looped impl
            # Reshape with the local number of experts as dim 0
            routed_input = routed_input.reshape(self.experts.num_local_experts, -1, dim)
        # routed_output shape (num_local_experts, tokens_per_expert * ep, dim) or
        # (sum(num_local_tokens_per_expert), dim) for the looped impl
        routed_output = self.experts(
            routed_input,
            num_tokens_per_expert=num_local_tokens_per_expert,
        )
        # routed_output shape (num_experts * tokens_per_expert, dim)
        routed_output = routed_output.reshape(-1, dim)

        # Gather the output from all experts (possibly with all-to-all)
        routed_output = self.token_dispatcher.token_unpermutation(
            routed_output,
            permute_metadata=permute_metadata,  # type: ignore[arg-type]
        )

        # Finish shared expert computation
        if self.shared_expert is not None:
            assert shared_expert_out is not None
            shared_expert_out = self.shared_expert.forward_output(shared_expert_out)
            out = shared_expert_out.reshape(-1, dim)
        else:
            out = torch.zeros_like(x.view(-1, dim))

        # Wait for all expert outputs to be available
        routed_output = self.token_dispatcher.wait_token_unpermutation(routed_output)

        # add experts output
        out.scatter_add_(dim=0, index=token_indices, src=routed_output)

        # possibly reduce
        if self.tp_gang:
            out = reduce(out, self.tp_gang)

        # shape (bs, slen, dim)
        out = out.reshape(bs, slen, dim)
        return out

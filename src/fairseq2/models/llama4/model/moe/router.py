# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from fairseq2.typing import DataType, Device


# TODO: update it with a formal AC check
def check_ac_recompute() -> bool:
    """Check if the current computation is during AC recomputation.

    Returns:
        bool: True if the current computation is during AC recomputation, False otherwise.
    """
    return torch._C._current_graph_task_id() != -1


class RouterOutput(NamedTuple):
    """Router output for Token Choice and Expert Choice routing.

    routed_input (torch.Tensor): tokens grouped together by experts indices with shape
        ``(bs*slen*experts_per_token, dim)`` for Token Choice, ``(num_experts*tokens_per_expert, dim)`` for Expert Choice.
    token_indices (torch.Tensor): token indices for routed_input with shape
        ``(bs*slen*experts_per_token,)`` for Token Choice, ``(num_experts*tokens_per_expert,)`` for Expert Choice.
    num_local_tokens_per_expert (Optional[torch.Tensor]):
        Number of tokens assigned to each expert with shape ``(num_experts,)`` if using Token Choice.
        None if using Expert Choice since tokens_per_expert is fixed across experts.
    """

    routed_input: torch.Tensor
    token_indices: torch.Tensor
    num_local_tokens_per_expert: Optional[torch.Tensor]


class Router(nn.Module):
    """This class implements experts choice routing. Each experts will select it's top K tokens based on
        the router scores. Refer to more details in https://arxiv.org/abs/2202.09368

    The router is never sharded.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        model_dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        capacity_factor (float): Capacity factor determines how many tokens each expert can choose.
            expert capacity = (number of tokens * capacity factor) / number of experts.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    gate: nn.Parameter
    model_dim: int
    num_experts: int
    top_k: int
    eval_with_expert_activation_model: bool
    expert_act_threshold: float

    def __init__(
        self,
        *,
        model_dim: int,
        num_experts: int,
        top_k: int,
        expert_act_threshold: float = 0.0,
        is_tp_sharding_strategy: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ):
        super().__init__()
        # Router is never sharded
        self.gate = nn.Parameter(
            torch.empty(model_dim, num_experts, device=device, dtype=dtype)
        )

        self.expert_act_threshold = expert_act_threshold

        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, x: torch.Tensor, experts_threshold: Optional[torch.Tensor] = None
    ) -> RouterOutput:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, model_dim)``.
            experts_threshold (Optional[torch.Tensor]): Threshold for each expert to determine whether to select a token.
                Shape ``(num_experts,)``.

        Returns:
            routed_input (torch.Tensor): input tokens grouped together by experts indices with shape
                ``(num_experts*tokens_per_expert, model_dim)``.
            token_indices (torch.Tensor): token indices for routed_input. Shape ``(num_experts*tokens_per_expert,)``.
            num_local_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)`` if using Token Choice.
                None if using Expert Choice in non looped_impl since tokens_per_expert is fixed across experts.
        """
        bszseqlen, dim = x.shape

        # logits shape (bs*slen, num_experts)
        logits = x @ self.gate

        scores, token_indices = torch.topk(logits, self.top_k, dim=1)

        scores = (
            torch.full_like(logits, float("-inf"))
            .scatter_(1, token_indices, scores)
            .transpose(0, 1)
        )

        token_indices = (
            torch.arange(bszseqlen, device=x.device)
            .view(1, -1)
            .expand(scores.size(0), -1)
        )

        scores = torch.sigmoid(scores)

        return RouterOutput(
            scores,
            token_indices,
            None,
        )

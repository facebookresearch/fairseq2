# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import Module, SiLU

from fairseq2.nn.grouped_projection import BatchLinear, GroupedProjection


class Experts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim (int): Input dimension.
        inner_dim (int): Hidden dimension.
        num_local_experts (int): Number of local experts in this grouped experts layer. Default is 1.
    """

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        num_local_experts: int = 1,
        activation: Module | None = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.num_local_experts = num_local_experts

        self.create_layers()

        self.activation: Module
        if activation is None:
            self.activation = SiLU()
        else:
            self.activation = activation

    def create_layers(self) -> None:
        """
        TODO(mgleize): For finetuning
        We intentionally fold the expert weights' ``num_local_experts`` dim-0
        into dim-1 at construction time and unfold them during forward for
        improved efficiency with FSDPv2, which does dim-0 per-parameter
        sharding. Without this, we would have highly uneven sharding across DP
        since local ``num_local_experts`` is typically much smaller than DP size.
        """
        self.gate_proj: GroupedProjection = BatchLinear(
            self.num_local_experts,
            self.model_dim,
            self.inner_dim,
        )
        self.inner_proj: GroupedProjection = BatchLinear(
            self.num_local_experts,
            self.model_dim,
            self.inner_dim,
        )
        self.output_proj: GroupedProjection = BatchLinear(
            self.num_local_experts,
            self.inner_dim,
            self.model_dim,
        )

    def forward(self, x: Tensor, *, num_tokens_per_expert: Tensor | None) -> Tensor:
        """
        Args:
            x (Tensor): tensor with shape ``(num_local_experts, tokens_per_expert, dim)``.
            num_tokens_per_expert (Optional[Tensor]): tensor with shape ``(num_local_experts,)``.
        Returns:
            torch.Tensor: Tensor with shape (num_local_experts, tokens_per_expert, dim)
        """
        # x shape (num_local_experts, tokens_per_expert, dim)
        h: Tensor = self.activation(self.gate_proj(x, num_tokens_per_expert))

        if self.inner_proj is not None:
            h = h * self.inner_proj(x, num_tokens_per_expert)
        # out shape (num_local_experts, tokens_per_expert, hidden_dim)
        out: Tensor = self.output_proj(h, num_tokens_per_expert)

        return out

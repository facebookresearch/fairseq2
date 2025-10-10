# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module, SiLU
from torch.nn.parameter import Parameter

from fairseq2.data_type import DataType
from fairseq2.device import META_DEVICE, Device
from fairseq2.gang import Gang
from fairseq2.nn.utils.module import to_empty
from fairseq2.ops.tensor_parallel import reduce, reduce_on_backward


def create_expert_glu_layers(
    module: Module,
    num_local_experts: int,
    model_dim: int,
    inner_dim: int,
    activation: Module | None = None,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> None:
    """Create GLU-based expert layers as attributes on the given module.

    :param module:
        The module to add the layers to.
    :param num_local_experts:
        The number of local experts.
    :param model_dim:
        The input and output dimension.
    :param inner_dim:
        The inner dimension.
    :param activation:
        The activation to apply after the gate projection.
    :param device:
        The device on which to initialize the layers.
    :param dtype:
        The data type of the layers.
    """
    # The expert dimension is folded with the first of model/inner dim
    # to optimize for FSDPv2 (which shards params on dim 0).
    module.gate_proj = Parameter(
        torch.empty(
            (
                num_local_experts * model_dim,
                inner_dim,
            ),
            device=device,
            dtype=dtype,
        )
    )
    module.inner_proj = Parameter(
        torch.empty(
            (
                num_local_experts * model_dim,
                inner_dim,
            ),
            device=device,
            dtype=dtype,
        )
    )
    module.output_proj = Parameter(
        torch.empty(
            (
                num_local_experts * inner_dim,
                model_dim,
            ),
            device=device,
            dtype=dtype,
        )
    )

    if activation is None:
        module.activation = SiLU()
    else:
        module.activation = activation


def _forward_glu_bmm_with_folded_experts(
    x: Tensor,
    gate_proj: Tensor,
    inner_proj: Tensor,
    output_proj: Tensor,
    num_local_experts: int,
    activation: Module,
) -> Tensor:
    """
    Forward pass for GLU-based experts with expert dimension
    folded into dim 0 in the weight tensors."""
    # unfold dim 0 on weights
    gate_proj = gate_proj.view(num_local_experts, -1, gate_proj.shape[-1])
    inner_proj = inner_proj.view(num_local_experts, -1, inner_proj.shape[-1])
    output_proj = output_proj.view(num_local_experts, -1, output_proj.shape[-1])

    # (num_local_experts, tokens_per_expert, dim)
    h: Tensor = activation(torch.bmm(x, gate_proj))

    h = h * torch.bmm(x, inner_proj)

    # (num_local_experts, tokens_per_expert, dim)
    out = torch.bmm(h, output_proj)

    return out


class ExpertNetwork(Module, ABC):
    def __init__(
        self,
        num_experts: int,
        model_dim: int,
        inner_dim: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.model_dim = model_dim
        self.inner_dim = inner_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape (num_local_experts, tokens_per_expert, dim).
        :returns: Output tensor of shape (num_local_experts, tokens_per_expert, dim).
        """

    @property
    def num_local_experts(self) -> int:
        """
        The number of experts local to the current process.
        Can be overriden in expert-parallel implementations.
        """
        return self.num_experts

    def extra_repr(self) -> str:
        """:meta private:"""
        local_s = (
            f"num_local_experts={self.num_local_experts}, "
            if self.num_local_experts != self.num_experts
            else ""
        )
        return f"num_experts={self.num_experts}, {local_s}model_dim={self.model_dim}, inner_dim={self.inner_dim}"

    if TYPE_CHECKING:
        __call__ = forward


@final
class GroupedExpertNetwork(ExpertNetwork):
    """This class implements a grouped experts layer as used in Mixture of Experts.
    Each expert is a variant of the Gated Linear Units network.
    See more details in https://arxiv.org/pdf/2002.05202.
    """

    gate_proj: Parameter
    inner_proj: Parameter
    output_proj: Parameter
    activation: Module

    def __init__(
        self,
        num_experts: int,
        model_dim: int,
        inner_dim: int,
        *,
        activation: Module | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ):
        """
        :param num_experts:
            The number of experts.
        :param model_dim:
            The input and output dimension.
        :param inner_dim:
            The inner dimension.
        :param activation:
            The activation to apply after the gate projection.
        """
        super().__init__(num_experts, model_dim, inner_dim)

        create_expert_glu_layers(
            self,
            num_experts,
            model_dim,
            inner_dim,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape (num_local_experts, tokens_per_expert, dim).
        :returns: Output tensor of shape (num_local_experts, tokens_per_expert, dim).
        """
        return _forward_glu_bmm_with_folded_experts(
            x,
            self.gate_proj,
            self.inner_proj,
            self.output_proj,
            self.num_local_experts,
            self.activation,
        )


@final
class TPShardedExpertNetwork(ExpertNetwork):
    """
    This class implements grouped experts sharded in one tensor-parallel dimension only.
    """

    gate_proj: Parameter
    inner_proj: Parameter
    output_proj: Parameter
    activation: Module

    @staticmethod
    def from_grouped_expert_network(
        experts: GroupedExpertNetwork,
        gang: Gang,
        reduce_output: bool = True,
    ) -> TPShardedExpertNetwork:
        """
        :param experts:
            The ``ExpertNetwork`` to shard.
        :param gang:
            The gang over which to shard the weight tensors.
        :param reduce_output:
            If ``False``, output will not be reduced.
        :returns:
            The sharded expert network.
        """
        device = experts.gate_proj.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `experts` must either match `gang.device` or must be of type `meta`."
            )

        sharded = TPShardedExpertNetwork(
            gang,
            experts.num_experts,
            experts.model_dim,
            experts.inner_dim,
            activation=experts.activation,
            reduce_output=reduce_output,
            device=META_DEVICE,
            dtype=experts.gate_proj.dtype,
        )

        if device.type != "meta":
            to_empty(sharded, device)

        sharded._copy_weight(experts.gate_proj, sharded.gate_proj, dim=2)
        sharded._copy_weight(experts.inner_proj, sharded.inner_proj, dim=2)
        sharded._copy_weight(experts.output_proj, sharded.output_proj, dim=1)

        return sharded

    def __init__(
        self,
        gang: Gang,
        num_experts: int,
        model_dim: int,
        inner_dim: int,
        *,
        activation: Module | None = None,
        reduce_output: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ):
        """
        :param gang:
            The gang over which to shard the weight tensors.
        :param num_experts:
            The number of experts.
        :param model_dim:
            The input and output dimension.
        :param inner_dim:
            The inner dimension.
        :param activation:
            The activation to apply after the gate projection.
        :param reduce_output:
            If ``False``, output will not be reduced.
        """
        super().__init__(num_experts, model_dim, inner_dim)

        if inner_dim % gang.size != 0:
            raise ValueError(
                f"`inner_dim` must be divisible by `gang.size` ({gang.size}), but is {inner_dim} instead."
            )

        self.gang = gang
        self.sharded_inner_dim = inner_dim // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either match `gang.device` or must be of type `meta`."
            )

        create_expert_glu_layers(
            self,
            num_experts,
            model_dim,
            self.sharded_inner_dim,
            activation=activation,
            device=device,
            dtype=dtype,
        )

        self.reduce_output = reduce_output

    def _copy_weight(
        self, source_param: Parameter, target_param: Parameter, dim: int
    ) -> None:
        with torch.no_grad():
            w = source_param.view(self.num_experts, -1, source_param.shape[-1])
            weight_shards = w.split(self.sharded_inner_dim, dim=dim)
            target_param.copy_(
                weight_shards[self.gang.rank].reshape(-1, target_param.shape[-1])
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape (num_local_experts, tokens_per_expert, dim).
        :returns: Output tensor of shape (num_local_experts, tokens_per_expert, dim).
        """
        x = reduce_on_backward(x, self.gang)

        out = _forward_glu_bmm_with_folded_experts(
            x,
            self.gate_proj,
            self.inner_proj,
            self.output_proj,
            self.num_local_experts,
            self.activation,
        )

        if self.reduce_output:
            out = reduce(out, self.gang)

        return out

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"inner_dim={self.sharded_inner_dim}"

        s = f"num_experts={self.num_experts}, model_dim={self.model_dim}, {s}"

        s = (
            f"{s}, "
            f"reduce_output={self.reduce_output}, "
            f"rank={self.gang.rank}, "
            f"world_size={self.gang.size}"
        )

        return s

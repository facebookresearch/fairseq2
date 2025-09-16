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
from torch.nn import Module
from torch.nn.parameter import Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import META_DEVICE, Device
from fairseq2.gang import Gang
from fairseq2.nn.utils.module import to_empty
from fairseq2.ops.tensor_parallel import gather, reduce, reduce_on_backward, scatter


class GroupedProjection(Module, ABC):
    """
    Applies a grouped linear transformation to incoming data.
    Alsp known as grouped GEMMs or batch matrix-matrix product (BMM).
    """

    def __init__(self, group_dim: int, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.group_dim = group_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, num_inputs_per_group: Tensor | None) -> Tensor:
        """
        :param x:
            The input to project. *Shape:* :math:`(G,*,H_{inp})`, where
            :math:`G` is the number of groups, and
            :math:`H_{inp}` is the input dimensionality.
        :param num_inputs_per_group:
            The number of inputs per group. *Shape:* :math:`(G)`, where
            :math:`G` is the number of groups.
            If not ``None``, a looped implementation is used instead of BMM,
            using the provided respected number of inputs for each group.

        :returns:
            The projected output. *Shape:* :math:`(*,H_{out})`, where all but
            the last dimension are the same shape as the input and
            :math:`H_{out}` is the output dimensionality.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"group_dim={self.group_dim}, input_dim={self.input_dim}, output_dim={self.output_dim}"

    if TYPE_CHECKING:
        __call__ = forward


def _maybe_bmm(
    x: Tensor,
    weight: Tensor,
    num_inputs_per_group: Tensor | None,
) -> Tensor:
    # Batch GEMM
    if num_inputs_per_group is None:
        return torch.bmm(x, weight)

    # Loop-based GEMM
    cum_size = num_inputs_per_group.cumsum(0, dtype=torch.long)

    buffer = torch.empty(
        int(cum_size[-1].item()),
        weight.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )

    start = 0
    for expert_index, end in enumerate(cum_size):
        torch.matmul(
            x[start:end],
            weight[expert_index],
            out=buffer[start:end],
        )
        start = end

    return buffer


@final
class BatchLinear(GroupedProjection):
    def __init__(
        self,
        extra_first_dim: int,
        input_dim: int,
        output_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param extra_first_dim:
            The size of the extra first dimension (for example: num_experts).
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        """
        super().__init__(extra_first_dim, input_dim, output_dim)

        self.weight = Parameter(
            torch.empty(
                (
                    extra_first_dim,
                    input_dim,
                    output_dim,
                ),
                device=device,
                dtype=dtype,
            )
        )

    @override
    def forward(self, x: Tensor, num_inputs_per_group: Tensor | None) -> Tensor:
        return _maybe_bmm(x, self.weight, num_inputs_per_group)


@final
class BatchColumnShardedLinear(GroupedProjection):
    """Represents a batched 3D tensor multiplication that is sharded across its output dimension."""

    @staticmethod
    def from_batch_linear(
        linear: BatchLinear, gang: Gang, *, gather_output: bool = True
    ) -> "BatchColumnShardedLinear":
        """Construct a :class:`BatchedColumnShardedLinear` by sharding a BatchedLinear.

        :param linear:
            The BatchedLinear to shard along last dimension.
        :param gang:
            The gang over which to shard the linear.
        :param gather_output:
            If ``True``, gather the sharded output into a single tensor.
        """
        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `linear` must either match `gang.device` or must be of type `meta`."
            )

        extra_first_dim, input_dim, output_dim = linear.weight.shape

        sharded = BatchColumnShardedLinear(
            gang,
            extra_first_dim,
            input_dim,
            output_dim,
            gather_output=gather_output,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded, device)

        sharded._copy_weight(linear.weight)

        return sharded

    def __init__(
        self,
        gang: Gang,
        extra_first_dim: int,
        input_dim: int,
        output_dim: int,
        *,
        gather_output: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang:
            The gang over which to shard the weight tensor.
        :param extra_first_dim:
            The size of the extra first dimension of the input tensor.
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        :param gather_output:
            If ``True``, gather the sharded output into a single tensor.
        """
        super().__init__(extra_first_dim, input_dim, output_dim)

        if output_dim % gang.size != 0:
            raise ValueError(
                f"`output_dim` must be divisible by `gang.size` ({gang.size}), but is {output_dim} instead."
            )

        self.gang = gang
        self.sharded_output_dim = output_dim // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either match `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (extra_first_dim, input_dim, self.sharded_output_dim),
            device=device,
            dtype=dtype,
        )

        self.weight = Parameter(weight)
        self.gather_output = gather_output

    def _copy_weight(self, param: Parameter) -> None:
        with torch.no_grad():
            weight_shards = param.split(self.sharded_output_dim, dim=-1)
            self.weight.copy_(weight_shards[self.gang.rank])

    @override
    def forward(self, x: Tensor, num_inputs_per_group: Tensor | None) -> Tensor:
        """Expected shape for x: (group_dim, N, input_dim)"""
        x = reduce_on_backward(x, self.gang)

        # Use torch.bmm for batched matrix multiplication
        x = _maybe_bmm(x, self.weight, num_inputs_per_group)

        if self.gather_output:
            x = gather(x, self.gang, dim=-1)

        # [group_dim, N, output_dim]
        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"group_dim={self.group_dim}, input_dim={self.input_dim}"

        if self.gather_output:
            s = f"{s}, output_dim={self.output_dim}"
        else:
            s = f"{s}, output_dim={self.sharded_output_dim}"

        s = (
            f"{s}, "
            f"gather_output={self.gather_output}, "
            f"rank={self.gang.rank}, "
            f"world_size={self.gang.size}"
        )

        return s


class BatchRowShardedLinear(GroupedProjection):
    """Represents a batched 3D tensor multiplication that is sharded across its input dimension."""

    @staticmethod
    def from_batch_linear(
        linear: BatchLinear,
        gang: Gang,
        scatter_input: bool = False,
        reduce_output: bool = True,
    ) -> "BatchRowShardedLinear":
        """Construct a :class:`BatchedRowShardedLinear` by sharding a BatchedLinear.

        :param linear:
            The BatchedLinear to shard along middle dimension.
        :param gang:
            The gang over which to shard the linear.
        :param scatter_input:
            If ``True``, inputs are considered already sharded and won't be
            scattered.
        :param reduce_output:
            If ``False``, output will not be reduced.
        """
        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `linear` must either match `gang.device` or must be of type `meta`."
            )

        extra_first_dim, input_dim, output_dim = linear.weight.shape

        sharded = BatchRowShardedLinear(
            gang,
            extra_first_dim,
            input_dim,
            output_dim,
            scatter_input=scatter_input,
            reduce_output=reduce_output,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded, device)

        sharded._copy_weight(linear.weight)

        return sharded

    def __init__(
        self,
        gang: Gang,
        extra_first_dim: int,
        input_dim: int,
        output_dim: int,
        *,
        scatter_input: bool = True,
        reduce_output: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang:
            The gang over which to shard the weight tensor.
        :param extra_first_dim:
            The size of the extra first dimension of the input tensor.
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        :param scatter_input:
            If ``True``, scatters the input tensor; otherwise, considers it
            already sharded.
        :param reduce_output:
            If ``False``, output will not be reduced.
        """
        super().__init__(extra_first_dim, input_dim, output_dim)

        if input_dim % gang.size != 0:
            raise ValueError(
                f"`input_dim` must be divisible by `gang.size` ({gang.size}), but is {input_dim} instead."
            )

        self.gang = gang
        self.sharded_input_dim = input_dim // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either match `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (extra_first_dim, self.sharded_input_dim, output_dim),
            device=device,
            dtype=dtype,
        )

        self.weight = Parameter(weight)
        self.scatter_input = scatter_input
        self.reduce_output = reduce_output

    def _copy_weight(self, param: Parameter) -> None:
        with torch.no_grad():
            weight_shards = param.split(self.sharded_input_dim, dim=1)
            self.weight.copy_(weight_shards[self.gang.rank])

    @override
    def forward(self, x: Tensor, num_inputs_per_group: Tensor | None) -> Tensor:
        """Expected shape for x: (extra_first_dim, N, input_dim)"""
        if self.scatter_input:
            x = scatter(x, self.gang, dim=-1)

        x = _maybe_bmm(x, self.weight, num_inputs_per_group)

        if self.reduce_output:
            x = reduce(x, self.gang)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        if self.scatter_input:
            s = f"input_dim={self.input_dim}"
        else:
            s = f"input_dim={self.sharded_input_dim}"

        s = f"group_dim={self.group_dim}, {s}, output_dim={self.output_dim}, "

        s = (
            f"{s}, "
            f"output_dim={self.output_dim}, "
            f"scatter_input={self.scatter_input}, "
            f"reduce_output={self.reduce_output}, "
            f"rank={self.gang.rank}, "
            f"world_size={self.gang.size}"
        )

        return s

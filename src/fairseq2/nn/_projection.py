# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import linear
from torch.nn.parameter import Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import META_DEVICE, Device
from fairseq2.error import InternalError
from fairseq2.gang import Gang
from fairseq2.nn.utils.module import to_empty
from fairseq2.ops.tensor_parallel import gather, reduce, reduce_on_backward, scatter
from fairseq2.typing import get_name_or_self


class Projection(Module, ABC):
    """Applies a linear transformation to incoming data."""

    input_dim: int
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim

        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The input to project. *Shape:* :math:`(*,H_{inp})`, where
            :math:`H_{inp}` is the input dimensionality.

        :returns: The projected output. *Shape:* :math:`(*,H_{out})`, where all
            but the last dimension are the same shape as the input and
            :math:`H_{out}` is the output dimensionality.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class Linear(Projection):
    """
    Applies a linear transformation to incoming data using weights and bias.

    Unless overridden by a subclass, the weights and bias are initialized from
    :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
    :math:`k = \\frac{1}{\\text{input_dim}}`.

    .. note::
        This class is identical to :class:`torch.nn.Linear`.
    """

    weight: Parameter
    bias: Parameter | None
    init_fn: Callable[[Linear], None] | None

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param input_dim: The dimensionality of inputs.
        :param output_dim: The dimensionality of projected outputs.
        :param bias: If ``True``, learns an additive bias.
        :param init_fn: The callable to initialize the weight and bias.
        """
        super().__init__(input_dim, output_dim)

        self.weight = Parameter(
            torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        )

        if bias:
            bias_ = Parameter(torch.empty((output_dim,), device=device, dtype=dtype))
        else:
            bias_ = None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            _init_uniform(self.weight, self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        bias = self.bias is not None

        s = f"input_dim={self.input_dim}, output_dim={self.output_dim}, bias={bias}"

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class ColumnShardedLinear(Projection):
    """Represents a :class:`Linear` that is sharded across its output dimension."""

    gang: Gang
    sharded_output_dim: int
    gather_output: bool
    weight: Parameter
    bias: Parameter | None
    init_fn: Callable[[Linear], None] | None

    @staticmethod
    def from_linear(
        linear: Linear, gang: Gang, gather_output: bool = True
    ) -> ColumnShardedLinear:
        """
        Constructs a :class:`ColumnShardedLinear` by sharding ``linear``.

        :param linear: The projection to shard.
        :param gang: The gang over which to shard ``linear``.
        :param gather_output: If ``True``, gather the sharded output into a
            single tensor.
        """
        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `linear` must either be same as `gang.device` or must be of type `meta`."
            )

        sharded_linear = ColumnShardedLinear(
            gang,
            linear.input_dim,
            linear.output_dim,
            bias=linear.bias is not None,
            gather_output=gather_output,
            init_fn=linear.init_fn,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded_linear, device)

        sharded_linear._copy_weight(linear)

        return sharded_linear

    def __init__(
        self,
        gang: Gang,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        gather_output: bool = True,
        init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang: The gang over which to shard the weight tensor.
        :param input_dim: The dimensionality of inputs.
        :param output_dim: The dimensionality of projected outputs.
        :param bias: If ``True``, learns an additive bias.
        :param gather_output: If ``True``, gather the sharded output into a
            single tensor.
        :param init_fn: The callable to initialize the weight and bias.
        """
        super().__init__(input_dim, output_dim)

        if output_dim % gang.size != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `gang.size` ({gang.size}), but is {output_dim} instead."
            )

        self.gang = gang

        self.sharded_output_dim = output_dim // gang.size

        self.gather_output = gather_output

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either be same as `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (self.sharded_output_dim, input_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        if bias:
            bias_ = Parameter(
                torch.empty((self.sharded_output_dim,), device=device, dtype=dtype)
            )
        else:
            bias_ = None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear = self._linear_like(self.gang.device)

        self._copy_weight(linear)

    def _copy_weight(self, linear: Linear) -> None:
        with torch.no_grad():
            weight_shards = linear.weight.split(self.sharded_output_dim)

            weight = weight_shards[self.gang.rank]

            self.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                bias_shards = linear.bias.split(self.sharded_output_dim)

                bias = bias_shards[self.gang.rank]

                self.bias.copy_(bias, non_blocking=True)

    @staticmethod
    def _pre_load_state_dict_hook(
        module: Module, state_dict: dict[str, object], prefix: str, *args: Any
    ) -> None:
        if not isinstance(module, ColumnShardedLinear):
            raise InternalError(f"`module` is of type `{type(module)}`.")

        key = f"{prefix}weight"

        weight = state_dict.get(key)
        if weight is None or not isinstance(weight, Tensor):
            return

        if weight.size(0) == module._output_dim:
            with torch.no_grad():
                weight_shards = weight.split(module.sharded_output_dim)

                state_dict[key] = weight_shards[module.gang.rank]

        if module.bias is not None:
            key = f"{prefix}bias"

            bias = state_dict.get(key)
            if bias is None or not isinstance(bias, Tensor):
                return

            if bias.size(0) == module.output_dim:
                with torch.no_grad():
                    bias_shards = bias.split(module.sharded_output_dim)

                    state_dict[key] = bias_shards[module.gang.rank]

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = reduce_on_backward(x, self.gang)

        x = linear(x, self.weight, self.bias)

        if self.gather_output:
            x = gather(x, self.gang, dim=-1)

        return x

    def to_linear(self, device: Device | None = None) -> Linear:
        """Converts this instance to a :class:`Linear`."""
        linear = self._linear_like(device=META_DEVICE)

        to_empty(linear, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=0)

            linear.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                bias = gather(self.bias, self.gang, dim=0)

                linear.bias.copy_(bias, non_blocking=True)

        return linear

    def _linear_like(self, device: Device) -> Linear:
        return Linear(
            self.input_dim,
            self.output_dim,
            bias=self.bias is not None,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        bias = self.bias is not None

        if self.gather_output:
            s = f"output_dim={self.output_dim}"
        else:
            s = f"output_dim={self.sharded_output_dim}"

        s = (
            f"rank={self.gang.rank}, "
            f"world_size={self.gang.size}, "
            f"input_dim={self.input_dim}"
            f"{s}, "
            f"gather_output={self.gather_output}, "
            f"bias={bias}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class RowShardedLinear(Projection):
    """Represents a :class:`Linear` that is sharded across its input dimension."""

    gang: Gang
    sharded_input_dim: int
    scatter_input: bool
    weight: Parameter
    bias: Parameter | None
    init_fn: Callable[[Linear], None] | None

    @staticmethod
    def from_linear(
        linear: Linear, gang: Gang, scatter_input: bool = False
    ) -> RowShardedLinear:
        """
        Constructs a :class:`RowShardedLinear` by sharding ``linear``.

        :param linear: The projection to shard.
        :param gang: The gang over which to shard ``linear``.
        :param scatter_input: If ``True``, inputs are considered already sharded
            and won't be scattered.
        """
        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `linear` must either be same as `gang.device` or must be of type `meta`."
            )

        sharded_linear = RowShardedLinear(
            gang,
            linear.input_dim,
            linear.output_dim,
            bias=linear.bias is not None,
            scatter_input=scatter_input,
            init_fn=linear.init_fn,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded_linear, device)

        sharded_linear._copy_weight(linear)

        return sharded_linear

    def __init__(
        self,
        gang: Gang,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        scatter_input: bool = True,
        init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang: The gang over which to shard the weight tensor.
        :param input_dim: The dimensionality of inputs.
        :param output_dim: The dimensionality of projected outputs.
        :param bias: If ``True``, learns an additive bias.
        :param scatter_input: If ``True``, scatters the input tensor; otherwise,
            considers it already sharded.
        :param init_fn: The callable to initialize the weight and bias.
        """
        super().__init__(input_dim, output_dim)

        if input_dim % gang.size != 0:
            raise ValueError(
                f"`input_dim` must be a multiple of `gang.size` ({gang.size}), but is {input_dim} instead."
            )

        self.gang = gang

        self.sharded_input_dim = input_dim // gang.size

        self.scatter_input = scatter_input

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either be same as `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (output_dim, self.sharded_input_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        if bias:
            bias_ = Parameter(torch.empty((output_dim,), device=device, dtype=dtype))
        else:
            bias_ = None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear = self._linear_like(self.gang.device)

        self._copy_weight(linear)

    def _copy_weight(self, linear: Linear) -> None:
        with torch.no_grad():
            weight_shards = linear.weight.split(self.sharded_input_dim, dim=1)

            weight = weight_shards[self.gang.rank]

            self.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                self.bias.copy_(linear.bias, non_blocking=True)

    @staticmethod
    def _pre_load_state_dict_hook(
        module: Module, state_dict: dict[str, object], prefix: str, *args: Any
    ) -> None:
        if not isinstance(module, RowShardedLinear):
            raise InternalError(f"`module` is of type `{type(module)}`.")

        key = f"{prefix}weight"

        weight = state_dict.get(key)
        if weight is None or not isinstance(weight, Tensor):
            return

        if weight.size(1) == module.input_dim:
            with torch.no_grad():
                weight_shards = weight.split(module.sharded_input_dim, dim=1)

                state_dict[key] = weight_shards[module.gang.rank]

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self._scatter_input:
            x = scatter(x, self.gang, dim=-1)

        x = linear(x, self.weight)

        x = reduce(x, self.gang)

        if self.bias is not None:
            x = x + self.bias

        return x

    def to_linear(self, device: Device | None = None) -> Linear:
        """Converts this instance to a :class:`Linear`."""
        linear = self._linear_like(device=META_DEVICE)

        to_empty(linear, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=1)

            linear.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                linear.bias.copy_(self.bias, non_blocking=True)

        return linear

    def _linear_like(self, device: Device) -> Linear:
        return Linear(
            self.input_dim,
            self.output_dim,
            bias=self.bias is not None,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        bias = self.bias is not None

        if self._scatter_input:
            s = f"input_dim={self._input_dim}"
        else:
            s = f"input_dim={self._sharded_input_dim}"

        s = (
            f"rank={self.gang.rank}, "
            f"world_size={self.gang.size}, "
            f"{s}, "
            f"scatter_input={self.scatter_input}, "
            f"output_dim={self.output_dim}, "
            f"bias={bias}, "
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class TiedProjection(Projection):
    """
    Applies a linear transformation to incoming data using the weights and bias
    of another :class:`~torch.nn.Module` instance.
    """

    weight: Parameter
    bias: Parameter | None

    def __init__(self, weight: Parameter, bias: Parameter | None) -> None:
        """
        :param weight: The shared weights.
        :param bias: The shared bias.
        """
        super().__init__(input_dim=weight.size(1), output_dim=weight.size(0))

        self.weight = weight

        self.bias = bias

    @override
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


@final
class IdentityProjection(Projection):
    """
    Used to disable a projection layer without changing the module architecture.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(input_dim=dim, output_dim=dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x


def _init_uniform(weight: Tensor, bias: Tensor | None) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    if bias is not None:
        fan_in = weight.size(1)

        m = 1
        if weight.ndim > 2:
            for s in weight.shape[2:]:
                m *= s

        fan_in *= m

        # We do not calculate the true standard deviation of the uniform
        # distribution (i.e. multiply with sqrt(3)). See
        # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        nn.init.uniform_(bias, -bound, bound)


def init_bert_projection(proj: Linear) -> None:
    """Initializes ``proj`` as a projection to be used in BERT-like models."""
    nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)

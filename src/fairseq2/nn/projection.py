# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, cast, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import linear
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import META_DEVICE, Device, get_current_device
from fairseq2.error import InternalError
from fairseq2.gang import Gang, Gangs, create_fake_gangs
from fairseq2.nn.sharded import Sharded
from fairseq2.nn.utils.module import get_name_or_self, to_empty
from fairseq2.ops.tensor_parallel import gather, reduce, reduce_on_backward, scatter
from fairseq2.utils.warn import _warn_deprecated


class Projection(Module, ABC):
    """Applies a linear transformation to input data."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """:meta private:"""
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Projects the input data.

        ``x`` must be of shape :math:`(*,H_{inp})`, where :math:`H_{inp}` is the
        input dimensionality of this module.

        The projected output will be of shape :math:`(*,H_{out})`, where all but
        the last dimension are the same shape as ``x`` and :math:`H_{out}` is
        the output dimensionality of this module.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class Linear(Projection):
    """
    Represents the standard implementation of :class:`Projection`.

    .. note::

        This class is identical to :class:`torch.nn.Linear`.
    """

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
        Unless overridden by ``init_fn``, the weight and bias of this module are
        initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
        :math:`k = \\frac{1}{\\text{input_dim}}`.

        If ``init_fn`` is provided, it will be used to initialize the weight and
        bias in :meth:`reset_parameters`.
        """
        super().__init__(input_dim, output_dim)

        self.weight = Parameter(
            torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        )

        if bias:
            bias_ = Parameter(torch.empty((output_dim,), device=device, dtype=dtype))
        else:
            bias_ = None

        self.bias: Parameter | None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_linear(self)

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
class ColumnShardedLinear(Projection, Sharded):
    """Represents a :class:`Projection` sharded across its output dimension."""

    @staticmethod
    def from_linear(
        linear: Linear,
        gang: Gang | None = None,
        *,
        gangs: Gangs | None = None,
        gather_output: bool = True,
    ) -> ColumnShardedLinear:
        """
        Creates a :class:`ColumnShardedLinear` by sharding ``linear`` over its
        output dimension using ``gangs.tp``.

        If ``gangs`` is ``None``, acts like a regular :class:`Linear` module.

        If ``gather_output`` is ``True``, the sharded outputs of all ranks will
        be gathered into a single tensor.
        """
        if gang is not None:
            _warn_deprecated(
                "`gang` parameter of `ColumnShardedLinear.from_linear` is deprecated and will be removed in v0.14. Please use the `gangs` parameter instead."
            )

            if gangs is not None:
                raise ValueError(
                    "`gang` and `gangs` cannot be provided at the same time."
                )
        else:
            if gangs is None:
                gangs = create_fake_gangs(linear.weight.device)

            gang = gangs.tp

        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                f"Device of `linear` must match `gangs.tp.device` or must be of type `meta`, but is `{device}` instead."
            )

        sharded_linear = ColumnShardedLinear(
            linear.input_dim,
            linear.output_dim,
            bias=linear.bias is not None,
            gangs=gangs,
            gather_output=gather_output,
            init_fn=linear.init_fn,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
            _tp_gang=gang,
        )

        if device.type != "meta":
            to_empty(sharded_linear, device)

        sharded_linear._copy_from_linear(linear)

        return sharded_linear

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        gangs: Gangs | None = None,
        gather_output: bool = True,
        init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        _tp_gang: Gang | None = None,
    ) -> None:
        super().__init__(input_dim, output_dim)

        if gangs is None:
            gangs = create_fake_gangs()

        tp_gang = gangs.tp if _tp_gang is None else _tp_gang

        if output_dim % tp_gang.size != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `gangs.tp.size` ({tp_gang.size}), but is {output_dim} instead."
            )

        self.tp_gang = tp_gang

        self.sharded = tp_gang.size > 1

        self.sharded_output_dim = output_dim // tp_gang.size

        self.gather_output = gather_output

        if device is None:
            device = get_current_device()

        if device.type != "meta" and device != tp_gang.device:
            raise ValueError(
                "`device` must match `gangs.tp.device` or must be of type `meta`."
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

        self.bias: Parameter | None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.sharded:
            linear = self._linear_like(self.tp_gang.device)

            self._copy_from_linear(linear)
        else:
            _init_linear(self)

    def _copy_from_linear(self, linear: Linear) -> None:
        with torch.no_grad():
            weight_shards = linear.weight.split(self.sharded_output_dim, dim=0)

            weight = weight_shards[self.tp_gang.rank]

            self.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                bias_shards = linear.bias.split(self.sharded_output_dim, dim=0)

                bias = bias_shards[self.tp_gang.rank]

                self.bias.copy_(bias, non_blocking=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.sharded:
            return linear(x, self.weight, self.bias)

        x = reduce_on_backward(x, self.tp_gang)

        x = linear(x, self.weight, self.bias)

        if self.gather_output:
            x = gather(x, self.tp_gang, dim=-1)

        return x

    def to_linear(self, device: Device | None = None) -> Linear:
        """Unshards this instance to a :class:`Linear`."""
        linear = self._linear_like(device=META_DEVICE)

        to_empty(linear, device or self.tp_gang.device)

        with torch.no_grad():
            if self.sharded:
                weight = gather(self.weight, self.tp_gang, dim=0)
            else:
                weight = self.weight

            linear.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                if self.sharded:
                    bias = gather(self.bias, self.tp_gang, dim=0)
                else:
                    bias = self.bias

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
    def get_shard_dims(self) -> list[tuple[Parameter, int]]:
        if self.bias is None:
            return [(self.weight, 0)]
        else:
            return [(self.weight, 0), (self.bias, 0)]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        bias = self.bias is not None

        if self.gather_output:
            s = f"output_dim={self.output_dim}"
        else:
            s = f"output_dim={self.sharded_output_dim}"

        s = (
            f"tp_rank={self.tp_gang.rank}, "
            f"tp_size={self.tp_gang.size}, "
            f"input_dim={self.input_dim}, "
            f"{s}, "
            f"gather_output={self.gather_output}, "
            f"bias={bias}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class RowShardedLinear(Projection, Sharded):
    """Represents a :class:`Projection` sharded across its input dimension."""

    @staticmethod
    def from_linear(
        linear: Linear,
        gang: Gang | None = None,
        *,
        gangs: Gangs | None = None,
        scatter_input: bool = False,
        reduce_output: bool = True,
    ) -> RowShardedLinear:
        """
        Creates a :class:`RowShardedLinear` by sharding ``linear`` over its
        input dimension using ``gangs.tp``.

        If ``gangs`` is ``None``, acts like a regular :class:`Linear` module.

        If ``scatter_input`` is ``True``, the inputs on all ranks are considered
        already sharded and won't be scattered.

        If ``reduce_output`` is ``True``, the outputs of all ranks will be
        all-reduced into a single tensor.
        """
        if gang is not None:
            _warn_deprecated(
                "`gang` parameter of `RowShardedLinear.from_linear` is deprecated and will be removed in v0.14. Please use the `gangs` parameter instead."
            )

            if gangs is not None:
                raise ValueError(
                    "`gang` and `gangs` cannot be provided at the same time."
                )
        else:
            if gangs is None:
                gangs = create_fake_gangs(linear.weight.device)

            gang = gangs.tp

        device = linear.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                f"Device of `linear` must match `gang.device` or must be of type `meta`, but is `{device}` instead."
            )

        sharded_linear = RowShardedLinear(
            linear.input_dim,
            linear.output_dim,
            bias=linear.bias is not None,
            gangs=gangs,
            scatter_input=scatter_input,
            reduce_output=reduce_output,
            init_fn=linear.init_fn,
            device=META_DEVICE,
            dtype=linear.weight.dtype,
            _tp_gang=gang,
        )

        if device.type != "meta":
            to_empty(sharded_linear, device)

        sharded_linear._copy_from_linear(linear)

        return sharded_linear

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        gangs: Gangs | None = None,
        scatter_input: bool = True,
        reduce_output: bool = True,
        init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        _tp_gang: Gang | None = None,
    ) -> None:
        super().__init__(input_dim, output_dim)

        if gangs is None:
            gangs = create_fake_gangs()

        tp_gang = gangs.tp if _tp_gang is None else _tp_gang

        if input_dim % tp_gang.size != 0:
            raise ValueError(
                f"`input_dim` must be a multiple of `gangs.tp.size` ({tp_gang.size}), but is {input_dim} instead."
            )

        self.tp_gang = tp_gang

        self.sharded = tp_gang.size > 1

        self.sharded_input_dim = input_dim // tp_gang.size

        self.scatter_input = scatter_input
        self.reduce_output = reduce_output

        if device is None:
            device = get_current_device()

        if device.type != "meta" and device != tp_gang.device:
            raise ValueError(
                "`device` must match `gangs.tp.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (output_dim, self.sharded_input_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        if bias:
            bias_ = Parameter(torch.empty((output_dim,), device=device, dtype=dtype))
        else:
            bias_ = None

        self.bias: Parameter | None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.sharded:
            linear = self._linear_like(self.tp_gang.device)

            self._copy_from_linear(linear)
        else:
            _init_linear(self)

    def _copy_from_linear(self, linear: Linear) -> None:
        with torch.no_grad():
            weight_shards = linear.weight.split(self.sharded_input_dim, dim=1)

            weight = weight_shards[self.tp_gang.rank]

            self.weight.copy_(weight, non_blocking=True)

        if self.bias is not None:
            if linear.bias is None:
                raise InternalError("`linear.bias` is `None`.")

            with torch.no_grad():
                self.bias.copy_(linear.bias, non_blocking=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.sharded:
            return linear(x, self.weight, self.bias)

        if self.scatter_input:
            x = scatter(x, self.tp_gang, dim=-1)

        x = linear(x, self.weight)

        if self.reduce_output:
            x = reduce(x, self.tp_gang)

        if self.bias is not None:
            x = x + self.bias

        return x

    def to_linear(self, device: Device | None = None) -> Linear:
        """Unshards this instance to a :class:`Linear`."""
        linear = self._linear_like(device=META_DEVICE)

        to_empty(linear, device or self.tp_gang.device)

        with torch.no_grad():
            if self.sharded:
                weight = gather(self.weight, self.tp_gang, dim=1)
            else:
                weight = self.weight

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
    def get_shard_dims(self) -> list[tuple[Parameter, int]]:
        return [(self.weight, 1)]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        bias = self.bias is not None

        if self.scatter_input:
            s = f"input_dim={self.input_dim}"
        else:
            s = f"input_dim={self.sharded_input_dim}"

        s = (
            f"tp_rank={self.tp_gang.rank}, "
            f"tp_size={self.tp_gang.size}, "
            f"{s}, "
            f"scatter_input={self.scatter_input}, "
            f"reduce_output={self.reduce_output}, "
            f"output_dim={self.output_dim}, "
            f"bias={bias}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class TiedProjection(Projection):
    """
    Applies a linear transformation to input data using the weight and bias
    of another :class:`~torch.nn.Module` instance.
    """

    def __init__(self, weight: Parameter, bias: Parameter | None) -> None:
        super().__init__(input_dim=weight.size(1), output_dim=weight.size(0))

        self.weight = weight
        self.bias = bias

    @override
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


@final
class IdentityProjection(Projection):
    """Disables a projection without changing architecture."""

    def __init__(self, dim: int) -> None:
        super().__init__(input_dim=dim, output_dim=dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x


def _init_linear(proj: Projection) -> None:
    m = cast(Linear, proj)

    if m.init_fn is not None:
        m.init_fn(m)
    else:
        _init_uniform(m.weight, m.bias)


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

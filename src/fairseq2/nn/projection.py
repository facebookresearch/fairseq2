# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import linear
from torch.nn.parameter import Parameter

from fairseq2.typing import DataType, Device, finaloverride, override


class Projection(Module, ABC):
    """Applies a linear transformation to incoming data."""

    input_dim: int
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        """
        super().__init__()

        self.input_dim, self.output_dim = input_dim, output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to project. *Shape:* :math:`(*,H_{inp})`, where
            :math:`H_{inp}` is the input dimensionality.

        :returns:
            The projected output. *Shape:* :math:`(*,H_{out})`, where all but
            the last dimension are the same shape as the input and
            :math:`H_{out}` is the output dimensionality.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


class Linear(Projection):
    """Applies a linear transformation to incoming data using weights and bias.

    Unless overridden by a subclass, the weights and bias are initialized from
    :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
    :math:`k = \\frac{1}{\\text{input_dim}}`.

    .. note::
        This class is identical to :class:`torch.nn.Linear`.
    """

    weight: Parameter
    bias: Optional[Parameter]
    skip_init: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        skip_init: bool = False,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        :param bias:
            If ``True``, learns an additive bias.
        :param skip_init:
            If ``True``, the weights and bias will be left uninitialized, and
            :meth:`reset_parameters` will become noop.

            This parameter is intended to be used by module authors who want to
            use a different weight initialization method in their modules.
        """
        super().__init__(input_dim, output_dim)

        self.weight = Parameter(
            torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        )

        if bias:
            self.bias = Parameter(
                torch.empty((output_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.skip_init = skip_init

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if not self.skip_init:
            self._do_reset_parameters()

    def _do_reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # We do not calculate the true standard deviation of the uniform
            # distribution (i.e. multiply with sqrt(3)). See
            # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
            bound = 1 / math.sqrt(self.input_dim) if self.input_dim > 0 else 0

            nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", bias={self.bias is not None}"


@final
class TiedProjection(Projection):
    """Applies a linear transformation to incoming data using the weights and
    bias of another :class:`~torch.nn.Module` instance."""

    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, weight: Parameter, bias: Optional[Parameter] = None) -> None:
        """
        :param weight:
            The shared weights.
        :param bias:
            The shared bias.
        """
        super().__init__(weight.size(1), weight.size(0))

        self.weight = weight
        self.bias = bias

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

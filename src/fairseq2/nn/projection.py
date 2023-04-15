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
import torch.nn.functional as F
from overrides import final as finaloverride
from overrides import override
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class Projection(Module, ABC):
    """Applies a linear transformation to incoming data."""

    inp_dim: int
    out_dim: int

    def __init__(self, inp_dim: int, out_dim: int) -> None:
        """
        :param inp_dim:
            The dimensionality of inputs.
        :param out_dim:
            The dimensionality of outputs.
        """
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to project. *Shape:* :math:`(*,H_{inp})`, where
            :math:`H_{inp}` is the input size.

        :returns:
            The projected output of ``x``. *Shape:* :math:`(*,H_{out})`, where
            all but the last dimension are the same shape as the input and
            :math:`H_{out}` is the output size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"inp_dim={self.inp_dim}, out_dim={self.out_dim}"


class ResettableProjection(Projection):
    """Applies a linear transformation to incoming data using weights and bias
    that can be re-initialized by calling :meth:`reset_parameters`."""

    weight: Parameter
    bias: Optional[Parameter]

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param inp_dim:
            The dimensionality of inputs.
        :param out_dim:
            The dimensionality of outputs.
        :param bias:
            If ``True``, learns an additive bias.
        """
        super().__init__(inp_dim, out_dim)

        self.weight = Parameter(
            torch.empty((out_dim, inp_dim), device=device, dtype=dtype)
        )

        if bias:
            self.bias = Parameter(torch.empty((out_dim,), device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", bias={self.bias is not None}"


@final
class Linear(ResettableProjection):
    """Applies a linear transformation to incoming data using weights and bias
    initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
    :math:`k = \\frac{1}{\\text{inp_dim}}`.

    .. note::
        This class is identical to :class:`torch.nn.Linear`.
    """

    @finaloverride
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # We do not calculate the true standard deviation of the uniform
            # distribution (i.e. multiply with sqrt(3)). See
            # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
            bound = 1 / math.sqrt(self.inp_dim) if self.inp_dim > 0 else 0

            nn.init.uniform_(self.bias, -bound, bound)


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
        super().__init__(weight.size(0), weight.size(1))

        self.weight = weight
        self.bias = bias

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

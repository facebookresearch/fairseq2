# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter


class Projection(Module, ABC):
    """Applies a linear projection to incoming data.

    :param inp_dim:
        The dimensionality of the inputs.
    :param out_dim:
        The dimensionality of the outputs.
    """

    inp_dim: int
    """The dimensionality of the inputs."""

    out_dim: int
    """The dimensionality of the outputs."""

    def __init__(self, inp_dim: int, out_dim: int) -> None:
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input. *Shape:* :math:`(*,H_{inp})`, where :math:`H_{inp}` is
            the input size.

        :returns:
            The output. *Shape:* :math:`(*,H_{out})`, where all but the last
            dimension are the same shape as the input and :math:`H_{out}` is the
            output size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"inp_dim={self.inp_dim}, out_dim={self.out_dim}"


class LocalProjection(Projection):
    """Applies a linear projection to incoming data using an in-memory weight
    matrix.

    :param inp_dim:
        The dimensionality of the inputs.
    :param out_dim:
        The dimensionality of the outputs.
    :param bias:
        If ``True``, the module will learn an additive bias.

    .. note::
        This class is identical to :class:`torch.nn.Linear`.
    """

    weight: Parameter
    bias: Optional[Parameter]

    def __init__(
        self, inp_dim: int, out_dim: int, bias: bool = False, device=None, dtype=None
    ) -> None:
        fct_kwargs: Dict = {"device": device, "dtype": dtype}

        super().__init__(inp_dim, out_dim)

        self.weight = Parameter(torch.empty(out_dim, inp_dim, **fct_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_dim, **fct_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module.

        Both the weights and the bias will be initialized from
        :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
        :math:`k = \\frac{1}{\\text{inp_dim}}`.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # We do not calculate the true standard deviation of the uniform
            # distribution (i.e. multiply with sqrt(3)). See:
            # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575
            bound = 1 / math.sqrt(self.inp_dim) if self.inp_dim > 0 else 0

            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # override
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """:meta private:"""
        return super().extra_repr() + f", bias={self.bias is not None}"


@final
class SharedProjection(Projection):
    """Applies a linear projection to incoming data using the weights and bias
    of another :class:`~torch.nn.Module` instance.

    :param weight:
        The shared weights.
    :param bias:
        The shared bias.
    """

    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, weight: Parameter, bias: Optional[Parameter] = None) -> None:
        super().__init__(weight.size(0), weight.size(1))

        self.weight = weight
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:  # override
        """
        :param x:
            The input. *Shape:* :math:`(*,H_{inp})`, where :math:`H_{inp}` is
            the input size.

        :returns:
            The output. *Shape:* :math:`(*,H_{out})`, where all but the last
            dimension are the same shape as the input and :math:`H_{out}` is the
            output size.
        """
        return F.linear(x, self.weight, self.bias)

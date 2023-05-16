# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from overrides import override
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.nn.projection import Projection, ResettableProjection


class VectorQuantizer(Module, ABC):
    """Applies vector quantization to incoming data."""

    input_dim: int
    quantized_dim: int

    def __init__(self, input_dim: int, quantized_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param quantized_dim:
            The dimensionality of quantized vectors.
        """
        super().__init__()

        self.input_dim = input_dim
        self.quantized_dim = quantized_dim

    @abstractmethod
    def forward(self, x: Tensor) -> "VectorQuantizerOutput":
        pass


@dataclass
class VectorQuantizerOutput(ABC):
    """The output of a vector quantizer."""

    quantized: Tensor
    """The quantized output."""

    @abstractmethod
    def compute_loss(self) -> Tensor:
        """Compute the loss."""

    @abstractmethod
    def get_ids(self, num_groups: int) -> Tensor:
        pass


@final
class GumbelVectorQuantizer(VectorQuantizer):
    """Applies vector quantization to incoming data using Gumbel-Softmax."""

    input_dim: int
    quantized_dim: int
    num_vars: int
    num_groups: int
    combine_groups: bool
    min_temp: float
    max_temp: float
    temp_decay: float
    vars: Parameter
    weight_proj: Projection
    num_updates: Tensor

    def __init__(
        self,
        input_dim: int,
        quantized_dim: int,
        num_vars: int,
        num_groups: int,
        temperature: Tuple[float, float, float],
        combine_groups: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        :param input_dim:
            The dimensionality of inputs.
        :param quantized_dim:
            The dimensionality of quantized outputs.
        :param num_vars:
            number of quantized vectors per group
        :param num_groups:
            number of groups for vector quantization
        :param temperature:
            The temperature for training. A tuple of maximum temperature,
            minimum temperature, and decay factor.
        :param combine_groups:
            whether to use the vectors for all groups
        """
        super().__init__(input_dim, quantized_dim)

        if quantized_dim % num_groups != 0:
            raise ValueError(
                f"`quantized_dim` must be divisible by `num_groups` ({num_groups}), but is {quantized_dim} instead."
            )

        var_dim = quantized_dim // num_groups

        num_groups = num_groups if not combine_groups else 1

        self.input_dim = input_dim
        self.quantized_dim = quantized_dim
        self.num_vars = num_vars
        self.num_groups = num_groups
        self.combine_groups = combine_groups
        self.max_temp, self.min_temp, self.temp_decay = temperature

        num_total_vars = num_groups * num_vars

        self.vars = Parameter(
            torch.empty((1, num_total_vars, var_dim), device=device, dtype=dtype)
        )

        self.weight_proj = NormalLinear(
            self.input_dim, num_total_vars, bias=True, device=device, dtype=dtype
        )

        num_updates = torch.empty((), device="cpu", dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.vars)

        self.num_updates.zero_()

    @finaloverride
    def forward(self, x: Tensor) -> "GumbelVectorQuantizerOutput":
        current_temp = self._compute_current_temp()

        bsz, tsz, fsz = x.shape

        x = self.weight_proj(x)

        x = x.unflatten(-1, (self.num_groups, self.num_vars))

        k = x.argmax(-1, keepdim=True)

        hard_x = torch.zeros_like(x, dtype=torch.float32).scatter_(-1, k, 1.0)

        hard_probs = hard_x.mean(dim=0)
        #        hard_x = (
        #            x.new_zeros(*x.shape)
        #            .scatter_(-1, k.view(-1, 1), 1.0)
        #            .view(bsz * tsz, self.num_groups, -1)
        #        )
        #        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.num_groups, -1).float(), dim=-1
        ).mean(dim=0)

        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=current_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.num_groups, 1)  # type: ignore[assignment]

        cb = x

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.num_groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        return GumbelVectorQuantizerOutput(
            x,
            cb,
            self.num_vars,
            self.num_groups,
            code_perplexity,
            prob_perplexity,
            current_temp,
        )

    def _compute_current_temp(self) -> float:
        temp = self.max_temp * self.temp_decay ** self.num_updates.item()

        self.num_updates.add_(1)

        return max(temp, self.min_temp)


class NormalLinear(ResettableProjection):
    @override
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


@final
@dataclass
class GumbelVectorQuantizerOutput(VectorQuantizerOutput):
    cb: Tensor
    num_vars: int
    num_groups: int
    code_perplexity: Tensor
    prob_perplexity: Tensor
    temperature: float

    @finaloverride
    def compute_loss(self) -> Tensor:
        n = self.num_vars * self.num_groups

        return (n - self.prob_perplexity) / n  # type: ignore[no-any-return]

    @finaloverride
    def get_ids(self, num_groups: int) -> Tensor:
        batch_size, seq_len = self.quantized.shape[:2]

        cb = self.cb.view(batch_size * seq_len * self.num_groups, -1)

        targets = cb.argmax(dim=-1).view(batch_size, seq_len, self.num_groups)

        targets = targets[:, :, :num_groups].view(-1, num_groups)

        return targets.detach()

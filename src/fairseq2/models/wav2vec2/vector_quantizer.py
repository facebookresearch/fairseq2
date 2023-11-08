# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import gumbel_softmax

from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device, finaloverride


class VectorQuantizer(Module, ABC):
    """Quantizes incoming data in a differentiable way."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> "VectorQuantizerOutput":
        pass


@dataclass
class VectorQuantizerOutput(ABC):
    """Holds the output of a vector quantizer."""

    quantized_vectors: Tensor
    """The quantized vector output."""

    @abstractmethod
    def compute_loss(self) -> Tensor:
        """Compute the loss."""

    @abstractmethod
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        pass


@final
class GumbelVectorQuantizer(VectorQuantizer):
    """Quantizes incoming data using Gumbel-Softmax."""

    input_dim: int
    output_dim: int
    num_codebooks: int
    num_codebook_entries: int
    min_temp: float
    max_temp: float
    temp_decay: float
    entry_proj: Linear
    entries: Parameter
    num_updates: Tensor

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        *,
        codebook_sampling_temperature: Tuple[float, float, float],
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        :param num_codebooks:
            number of groups for vector quantization
        :param num_codebook_entries:
            number of quantized vectors per group
        :param codebook_sampling_temperature:
            The temperature for training. A tuple of maximum temperature,
            minimum temperature, and decay factor.
        """
        super().__init__(input_dim, output_dim)

        if output_dim % num_codebooks != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `num_codebooks` ({num_codebooks}), but is {output_dim} instead."
            )

        entry_dim = output_dim // num_codebooks

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries
        self.max_temp, self.min_temp, self.temp_decay = codebook_sampling_temperature

        num_total_entries = num_codebooks * num_codebook_entries

        self.entry_proj = Linear(
            self.input_dim,
            num_total_entries,
            bias=True,
            init_fn=init_entry_projection,
            device=device,
            dtype=dtype,
        )

        self.entries = Parameter(
            torch.empty((1, num_total_entries, entry_dim), device=device, dtype=dtype)
        )

        num_updates = torch.empty((), device=device, dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.entries)

        self.num_updates.zero_()

    @finaloverride
    def forward(self, x: Tensor) -> "GumbelVectorQuantizerOutput":
        current_temp = self._compute_current_temp()

        bsz, tsz, fsz = x.shape

        x = self.entry_proj(x)

        #        x = x.unflatten(-1, (self.num_codebooks, self.num_codebook_entries))
        #
        #        k = x.argmax(-1, keepdim=True)
        #
        #        hard_x = torch.zeros_like(x, dtype=torch.float32).scatter_(-1, k, 1.0)
        #
        #        hard_probs = hard_x.mean(dim=0)
        x = x.view(bsz * tsz * self.num_codebooks, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.num_codebooks, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)

        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.num_codebooks, -1).float(), dim=-1
        ).mean(dim=0)

        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x = gumbel_softmax(x.float(), tau=current_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        cb = x

        x = x.unsqueeze(-1) * self.entries
        x = x.view(bsz * tsz, self.num_codebooks, self.num_codebook_entries, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        return GumbelVectorQuantizerOutput(
            x,
            cb,
            self.num_codebooks,
            self.num_codebook_entries,
            code_perplexity,
            prob_perplexity,
            current_temp,
        )

    def _compute_current_temp(self) -> float:
        temp = self.max_temp * self.temp_decay ** int(self.num_updates)

        if self.training:
            self.num_updates.add_(1)

        return max(temp, self.min_temp)


def init_entry_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, mean=0.0, std=1.0)

    assert proj.bias is not None

    nn.init.zeros_(proj.bias)


@final
@dataclass
class GumbelVectorQuantizerOutput(VectorQuantizerOutput):
    cb: Tensor
    num_codebooks: int
    num_codebook_entries: int
    code_perplexity: Tensor
    prob_perplexity: Tensor
    temperature: float

    @finaloverride
    def compute_loss(self) -> Tensor:
        num_entries = self.num_codebooks * self.num_codebook_entries

        return (num_entries - self.prob_perplexity) / num_entries  # type: ignore[no-any-return]

    @finaloverride
    def get_target_indices(self, num_codebooks: int) -> Tensor:
        batch_size, seq_len = self.quantized_vectors.shape[:2]

        cb = self.cb.view(batch_size * seq_len * self.num_codebooks, -1)

        indices = cb.argmax(dim=-1).view(-1, self.num_codebooks)

        indices = indices[..., :num_codebooks]

        return indices.detach()

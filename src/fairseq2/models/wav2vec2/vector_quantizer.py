# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import gumbel_softmax
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import Linear


class Wav2Vec2VectorQuantizer(Module, ABC):
    """Quantizes input data in a differentiable way."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        codebook_sampling_temperature: tuple[float, float, float],
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries
        self.codebook_sampling_temperature = codebook_sampling_temperature

    @abstractmethod
    def forward(self, x: Tensor) -> Wav2Vec2VectorQuantizerOutput: ...

    if TYPE_CHECKING:
        __call__ = forward


@dataclass
class Wav2Vec2VectorQuantizerOutput:
    quantized_vectors: Tensor
    cb: Tensor
    code_perplexity: Tensor
    prob_perplexity: Tensor
    temperature: float


@final
class GumbelWav2Vec2VectorQuantizer(Wav2Vec2VectorQuantizer):
    """Quantizes input data using Gumbel-Softmax."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        codebook_sampling_temperature: tuple[float, float, float],
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ):
        """
        :param input_dim: The dimensionality of inputs.
        :param output_dim: The dimensionality of quantized outputs.
        :param num_codebooks: The number of groups for vector quantization.
        :param num_codebook_entries: The number of quantized vectors per group.
        :param codebook_sampling_temperature: The temperature for training. A
            tuple of maximum temperature, minimum temperature, and decay factor.
        """
        super().__init__(
            input_dim,
            output_dim,
            num_codebooks,
            num_codebook_entries,
            codebook_sampling_temperature,
        )

        if output_dim % num_codebooks != 0:
            raise ValueError(
                f"`output_dim` must be a multiple of `num_codebooks` ({num_codebooks}), but is {output_dim} instead."
            )

        num_total_entries = num_codebooks * num_codebook_entries

        self.entry_proj = Linear(
            input_dim,
            num_total_entries,
            bias=True,
            init_fn=_init_entry_projection,
            device=device,
            dtype=dtype,
        )

        entry_dim = output_dim // num_codebooks

        entries = torch.empty(
            (1, num_total_entries, entry_dim), device=device, dtype=dtype
        )

        self.entries = Parameter(entries)

        num_updates = torch.empty((), device=device, dtype=torch.int64)

        self.num_updates: Tensor

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.entries)

        self.num_updates.zero_()

    @override
    def forward(self, x: Tensor) -> Wav2Vec2VectorQuantizerOutput:
        temp = self._compute_current_temp()

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
            x = gumbel_softmax(x.float(), tau=temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        cb = x

        @torch.compile(fullgraph=True)
        def compute_sum(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(
                x.view(bsz * tsz, self.num_codebooks, self.num_codebook_entries, 1)
                * self.entries.view(
                    1, self.num_codebooks, self.num_codebook_entries, -1
                ),
                dim=-2,
            )

        x = compute_sum(x).view(bsz, tsz, -1)

        return Wav2Vec2VectorQuantizerOutput(
            x, cb, code_perplexity, prob_perplexity, temp
        )

    def _compute_current_temp(self) -> float:
        max_temp, min_temp, temp_decay = self.codebook_sampling_temperature

        temp = max_temp * temp_decay ** int(self.num_updates)

        if self.training:
            self.num_updates.add_(1)

        return max(temp, min_temp)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_codebooks={self.num_codebooks}, "
            f"num_codebook_entries={self.num_codebook_entries}, "
            f"codebook_sampling_temperature={self.codebook_sampling_temperature}"
        )


def _init_entry_projection(proj: Linear) -> None:
    nn.init.normal_(proj.weight, mean=0.0, std=1.0)

    if proj.bias is None:
        raise ValueError("`proj.bias` must not be `None`.")

    nn.init.zeros_(proj.bias)

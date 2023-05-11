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
from torch import Tensor
from torch.nn import Module


class VectorQuantizer(Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> "VectorQuantizerOutput":
        pass


class VectorQuantizerOutput(ABC):
    @abstractmethod
    def compute_loss(self) -> Tensor:
        pass

    @abstractmethod
    def get_ids(self, num_groups: int) -> Tensor:
        pass


@final
class GumbelVectorQuantizer(VectorQuantizer):
    num_updates: Tensor

    def __init__(
        self,
        dim: int,
        num_vars: int,
        num_groups: int,
        temperature: Tuple[float, float, float],
        combine_groups: bool,
        vq_dim: int,
        weight_proj_depth: int = 1,
        weight_proj_factor: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            num_groups: number of groups for vector quantization
            temperature: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.num_groups = num_groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars

        assert (
            vq_dim % num_groups == 0
        ), f"dim {vq_dim} must be divisible by groups {num_groups} for concatenation"

        var_dim = vq_dim // num_groups
        num_groups = num_groups if not combine_groups else 1

        self.vars = nn.Parameter(
            torch.empty((1, num_groups * num_vars, var_dim), device=device)
        )

        self.weight_proj = nn.Linear(
            self.input_dim, num_groups * num_vars, device=device
        )
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        self.max_temp, self.min_temp, self.temp_decay = temperature
        self.curr_temp = self.max_temp
        self.codebook_indices = None

        num_updates = torch.empty((1,), device="cpu", dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.vars)

        self.num_updates.zero_()

    #    def get_codebook_indices(self):
    #        if self.codebook_indices is None:
    #            from itertools import product
    #
    #            p = [range(self.num_vars)] * self.num_groups
    #            inds = list(product(*p))
    #            self.codebook_indices = torch.tensor(
    #                inds, dtype=torch.long, device=self.vars.device
    #            ).flatten()
    #
    #            if not self.combine_groups:
    #                self.codebook_indices = self.codebook_indices.view(
    #                    self.num_vars**self.num_groups, -1
    #                )
    #                for b in range(1, self.num_groups):
    #                    self.codebook_indices[:, b] += self.num_vars * b
    #                self.codebook_indices = self.codebook_indices.flatten()
    #        return self.codebook_indices
    #
    #    def codebook(self):
    #        indices = self.get_codebook_indices()
    #        return (
    #            self.vars.squeeze(0)
    #            .index_select(0, indices)
    #            .view(self.num_vars**self.num_groups, -1)
    #        )
    #
    #    def sample_from_codebook(self, b, n):
    #        indices = self.get_codebook_indices()
    #        indices = indices.view(-1, self.num_groups)
    #        cb_size = indices.size(0)
    #        assert (
    #            n < cb_size
    #        ), f"sample size {n} is greater than size of codebook {cb_size}"
    #        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
    #        indices = indices[sample_idx]
    #
    #        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
    #        return z
    #
    #    def to_codebook_index(self, indices):
    #        res = indices.new_full(indices.shape[:-1], 0)
    #        for i in range(self.num_groups):
    #            exponent = self.num_groups - i - 1
    #            res += indices[..., i] * (self.num_vars**exponent)
    #        return res

    @finaloverride
    def forward(self, x: Tensor) -> "GumbelVectorQuantizerOutput":
        self._compute_current_temp()

        #        result = {"num_vars": self.num_vars * self.num_groups}
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.num_groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.num_groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
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
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.num_groups, 1)  # type: ignore[assignment]

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.num_groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        return GumbelVectorQuantizerOutput(
            x,
            self.num_vars,
            self.num_groups,
            code_perplexity,
            prob_perplexity,
            self.curr_temp,
        )

    def _compute_current_temp(self) -> None:
        temp = self.max_temp * self.temp_decay ** self.num_updates.item()

        self.curr_temp = max(temp, self.min_temp)

        self.num_updates.add_(1)


@final
@dataclass
class GumbelVectorQuantizerOutput(VectorQuantizerOutput):
    x: Tensor
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
        batch_size, seq_len = self.x.shape[:2]

        x = self.x.view(batch_size * seq_len * self.num_groups, -1)

        targets = x.argmax(dim=-1).view(batch_size, seq_len, self.num_groups)

        return targets[:, :, :num_groups].detach()

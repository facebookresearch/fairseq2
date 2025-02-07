# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


class Sampler(ABC):
    """Represents a sampling algorithm."""

    @abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        """
        :param probs:
            The next-step probability of each vocabulary entry. *Shape:*
            :math:`(N,V)`, where :math:`N` is the batch size and :math:`V` is
            the size of the vocabulary.
        """


@final
class TopPSampler(Sampler):
    """Selects the next step randomly from the smallest set of candidates for
    which the cumulative probability exceeds a specified value p.

    Also known as Nucleus Sampling as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1904.09751`.
    """

    _p: float

    def __init__(self, p: float = 0.9) -> None:
        """
        :param p:
            The cumulative probability threshold.
        """
        self._p = p

    @override
    def sample(self, probs: Tensor) -> Tensor:
        # Previous operations in the generation like step processors might have
        # modified the probabilities. Normalize the distribution.
        probs = probs / probs.sum(dim=-1, keepdim=True)

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

        # (N, V)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = (cumsum_probs - sorted_probs) > self._p

        sorted_probs[mask] = 0.0

        # Normalize.
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        # (N, 1)
        indices = sorted_indices.gather(
            dim=-1, index=torch.multinomial(sorted_probs, num_samples=1)
        )

        # (N, 1) -> (N)
        return indices.squeeze(-1)  # type: ignore[no-any-return]


@final
class TopKSampler(Sampler):
    """Selects the next step randomly from the k mosty likely candidates."""

    _k: int

    def __init__(self, k: int) -> None:
        """
        :param k:
            The number of candidates to select from.
        """
        self._k = k

    @override
    def sample(self, probs: Tensor) -> Tensor:
        k = min(self._k, probs.size(1))

        if k == 1:
            # (N, 1)
            indices = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # (N, V) -> (N, K)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1, sorted=False)

            # Normalize.
            topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

            # (N, 1)
            indices = topk_indices.gather(
                dim=-1, index=torch.multinomial(topk_probs, num_samples=1)
            )

        # (N, 1) -> (N)
        return indices.squeeze(-1)


class SamplerHandler(ABC):
    @abstractmethod
    def create(self, config: object) -> Sampler: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class UnknownSamplerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sequence generator sampler.")

        self.name = name


TOP_P_SAMPLER: Final = "top_p"


@dataclass(kw_only=True)
class TopPSamplerConfig:
    p: float = 1.0


@final
class TopPSamplerHandler(SamplerHandler):
    @override
    def create(self, config: object) -> Sampler:
        config = structure(config, TopPSamplerConfig)

        validate(config)

        return TopPSampler(p=config.p)

    @property
    @override
    def config_kls(self) -> type[object]:
        return TopPSamplerConfig


TOP_K_SAMPLER: Final = "top_k"


@dataclass(kw_only=True)
class TopKSamplerConfig:
    k: int = 1


@final
class TopKSamplerHandler(SamplerHandler):
    @override
    def create(self, config: object) -> Sampler:
        config = structure(config, TopKSamplerConfig)

        validate(config)

        return TopKSampler(k=config.k)

    @property
    @override
    def config_kls(self) -> type[object]:
        return TopKSamplerConfig

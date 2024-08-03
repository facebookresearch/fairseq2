# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.generation.sampling.sampler import Sampler, TopKSampler, TopPSampler

if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    sampler_factories = ConfigBoundFactoryRegistry[[], Sampler]()
else:
    sampler_factories = ConfigBoundFactoryRegistry()

sampler_factory = sampler_factories.decorator


@dataclass
class TopPSamplerConfig:
    """Holds the configuration of a :class:`TopPSampler`."""

    p: float = 1.0
    """The cumulative probability threshold."""


@sampler_factory("top-p")
def create_top_p_sampler(config: TopPSamplerConfig) -> TopPSampler:
    return TopPSampler(p=config.p)


@dataclass
class TopKSamplerConfig:
    """Holds the configuration of a :class:`TopKSampler`."""

    k: int = 1
    """The number of candidates to select from."""


@sampler_factory("top-k")
def create_top_k_sampler(config: TopKSamplerConfig) -> TopKSampler:
    return TopKSampler(k=config.k)

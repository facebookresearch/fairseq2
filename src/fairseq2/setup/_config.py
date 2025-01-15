# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.generation import (
    AlgorithmSection,
    AlgorithmSectionHandler,
    BeamSearchAlgorithmHandler,
    SamplerHandler,
    SamplerSection,
    SamplerSectionHandler,
)
from fairseq2.utils.config import ConfigSectionHandler


def _register_config_sections(context: RuntimeContext) -> None:
    registry = context.get_registry(ConfigSectionHandler)

    handler: ConfigSectionHandler

    # Sampler
    sampler_handlers = context.get_registry(SamplerHandler)

    handler = SamplerSectionHandler(sampler_handlers)

    registry.register(SamplerSection, handler)

    # Algorithm
    algorithm_handlers = context.get_registry(BeamSearchAlgorithmHandler)

    handler = AlgorithmSectionHandler(algorithm_handlers)

    registry.register(AlgorithmSection, handler)

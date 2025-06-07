# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.gang import Gangs
from fairseq2.profilers import CompositeProfiler, NoopProfiler, Profiler, TorchProfiler
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import (
    CommonSection,
    TorchProfilerConfig,
    get_config_section,
    get_output_dir,
)
from fairseq2.utils.structured import StructureError


def create_profiler(resolver: DependencyResolver) -> Profiler:
    common_section = get_config_section(resolver, "common", CommonSection)

    profilers = []

    for name, config in common_section.profilers.items():
        try:
            profiler = resolve_component(resolver, Profiler, name, config)
        except StructureError as ex:
            raise StructureError(
                f"The '{name}' profiler configuration cannot be parsed. See the nested exception for details."
            ) from ex

        profilers.append(profiler)

    return CompositeProfiler(profilers)


def create_torch_profiler(
    resolver: DependencyResolver, config: TorchProfilerConfig
) -> Profiler:
    if not config.enabled:
        return NoopProfiler()

    gangs = resolver.resolve(Gangs)

    output_dir = get_output_dir(resolver)

    tb_dir = output_dir.joinpath("tb")

    return TorchProfiler(
        config.skip_n_steps,
        config.wait_n_steps,
        config.num_warmup_steps,
        config.num_active_steps,
        config.repeat,
        tb_dir,
        gangs,
    )

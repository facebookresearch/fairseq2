# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.profilers import CompositeProfiler, Profiler, TorchProfiler
from fairseq2.recipe.config import CommonSection, get_config_section, get_output_dir
from fairseq2.runtime.dependency import DependencyResolver


def _create_profiler(resolver: DependencyResolver) -> Profiler:
    profilers = []

    profiler = _maybe_create_torch_profiler(resolver)
    if profiler is not None:
        profilers.append(profiler)

    return CompositeProfiler(profilers)


def _maybe_create_torch_profiler(resolver: DependencyResolver) -> Profiler | None:
    common_section = get_config_section(resolver, "common", CommonSection)

    section = common_section.profilers.torch

    if not section.enabled:
        return None

    gangs = resolver.resolve(Gangs)

    output_dir = get_output_dir(resolver)

    tb_dir = output_dir.joinpath("tb")

    return TorchProfiler(
        section.skip_n_steps,
        section.wait_n_steps,
        section.num_warmup_steps,
        section.num_active_steps,
        section.repeat,
        tb_dir,
        gangs,
    )

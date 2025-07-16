# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from fairseq2.context import RuntimeContext
from fairseq2.gang import Gangs
from fairseq2.profilers import (
    CompositeProfiler,
    Profiler,
    ProfilerHandler,
    UnknownProfilerError,
)
from fairseq2.recipes.config import CommonSection
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError


def create_profiler(
    context: RuntimeContext,
    common_section: CommonSection,
    gangs: Gangs,
    output_dir: Path,
) -> Profiler:
    profiler_handlers = context.get_registry(ProfilerHandler)

    creator = _ProfilerCreator(profiler_handlers)

    return creator.create(common_section, gangs, output_dir)


@final
class _ProfilerCreator:
    _profiler_handlers: Provider[ProfilerHandler]

    def __init__(self, profiler_handlers: Provider[ProfilerHandler]) -> None:
        self._profiler_handlers = profiler_handlers

    def create(
        self, common_section: CommonSection, gangs: Gangs, output_dir: Path
    ) -> Profiler:
        profilers = []

        for profiler_name, profiler_config in common_section.profilers.items():
            try:
                handler = self._profiler_handlers.get(profiler_name)
            except LookupError:
                raise UnknownProfilerError(profiler_name) from None

            try:
                profiler = handler.create(profiler_config, gangs, output_dir)
            except StructureError as ex:
                raise StructureError(
                    f"`common.profilers.{profiler_name}.config` cannot be structured. See the nested exception for details."
                ) from ex

            profilers.append(profiler)

        return CompositeProfiler(profilers)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.datasets import DataReader
from fairseq2.gang import Gangs
from fairseq2.generator import BatchT, Generator, GeneratorUnit
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.profilers import Profiler
from fairseq2.recipe.config import GeneratorSection, get_config_section
from fairseq2.recipe.seed import SeedHolder
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.stopwatch import Stopwatch


def _create_generator(
    resolver: DependencyResolver,
    unit: GeneratorUnit[BatchT],
    data_reader: DataReader[BatchT],
) -> Generator:
    generator_section = get_config_section(resolver, "generator", GeneratorSection)

    seed_holder = resolver.resolve(SeedHolder)

    gangs = resolver.resolve(Gangs)

    metric_recorder = resolver.resolve(MetricRecorder)

    profiler = resolver.resolve(Profiler)

    device_stat_tracker = resolver.resolve(DeviceStatTracker)

    wall_watch = resolver.resolve(Stopwatch)

    progress_reporter = resolver.resolve(ProgressReporter)

    seed = seed_holder.advance()

    return Generator(
        unit=unit,
        data_reader=data_reader,
        gangs=gangs,
        dtype=generator_section.dtype,
        amp=generator_section.amp,
        seed=seed,
        metric_recorder=metric_recorder,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=wall_watch,
        progress_reporter=progress_reporter,
    )

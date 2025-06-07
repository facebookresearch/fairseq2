# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from fairseq2.datasets import DataReader
from fairseq2.evaluator import BatchT, Evaluator, EvalUnit
from fairseq2.gang import Gangs
from fairseq2.recipe.config import EvaluatorSection, get_config_section
from fairseq2.recipe.device import _create_device_stat_tracker
from fairseq2.recipe.metric_recorders import _create_metric_recorder
from fairseq2.recipe.profilers import _create_profiler
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import SeedHolder
from fairseq2.utils.stopwatch import Stopwatch


def _create_evaluator(
    resolver: DependencyResolver,
    units: Sequence[EvalUnit[BatchT]],
    data_readers: Sequence[DataReader[BatchT]],
) -> Evaluator:
    evaluator_section = get_config_section(resolver, "evaluator", EvaluatorSection)

    seed_holder = resolver.resolve(SeedHolder)

    gangs = resolver.resolve(Gangs)

    metric_recorder = _create_metric_recorder(resolver)

    profiler = _create_profiler(resolver)

    device_stat_tracker = _create_device_stat_tracker(resolver)

    wall_watch = resolver.resolve(Stopwatch)

    progress_reporter = resolver.resolve(ProgressReporter)

    seed = seed_holder.advance()

    return Evaluator(
        units=units,
        data_readers=data_readers,
        gangs=gangs,
        dtype=evaluator_section.dtype,
        amp=evaluator_section.amp,
        seed=seed,
        metric_recorder=metric_recorder,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=wall_watch,
        progress_reporter=progress_reporter,
    )

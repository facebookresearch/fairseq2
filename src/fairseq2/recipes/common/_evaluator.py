# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from fairseq2.context import RuntimeContext
from fairseq2.datasets import DataReader
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.gang import Gangs
from fairseq2.recipes import Evaluator, EvalUnit
from fairseq2.recipes.config import CommonSection, EvaluatorSection

# isort: split

from fairseq2.recipes.common._device import create_device_stat_tracker
from fairseq2.recipes.common._metrics import create_metric_recorder
from fairseq2.recipes.common._profilers import create_profiler

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


def create_evaluator(
    context: RuntimeContext,
    evaluator_section: EvaluatorSection,
    common_section: CommonSection,
    output_dir: Path,
    units: Sequence[EvalUnit[BatchT]],
    data_readers: Sequence[DataReader[BatchT]],
    gangs: Gangs,
    seed: int,
    *,
    hyper_params: object = None,
) -> Evaluator:
    metric_recorder = create_metric_recorder(
        context, common_section, gangs, output_dir, hyper_params
    )

    profiler = create_profiler(context, common_section, gangs, output_dir)

    device_stat_tracker = create_device_stat_tracker(gangs)

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
        wall_watch=context.wall_watch,
        progress_reporter=context.progress_reporter,
    )

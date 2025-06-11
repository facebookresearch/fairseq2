# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from fairseq2.context import RuntimeContext
from fairseq2.datasets import DataReader
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.gang import Gangs
from fairseq2.recipes import Generator, GeneratorUnit
from fairseq2.recipes.config import CommonSection, GeneratorSection

# isort: split

from fairseq2.recipes.common._device import create_device_stat_tracker
from fairseq2.recipes.common._metrics import create_metric_recorder
from fairseq2.recipes.common._profilers import create_profiler

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


def create_generator(
    context: RuntimeContext,
    generator_section: GeneratorSection,
    common_section: CommonSection,
    output_dir: Path,
    unit: GeneratorUnit[BatchT],
    data_reader: DataReader[BatchT],
    gangs: Gangs,
    seed: int,
    *,
    hyper_params: object = None,
) -> Generator:
    metric_recorder = create_metric_recorder(
        context, common_section, gangs, output_dir, hyper_params
    )

    profiler = create_profiler(context, common_section, gangs, output_dir)

    device_stat_tracker = create_device_stat_tracker(gangs)

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
        wall_watch=context.wall_watch,
        progress_reporter=context.progress_reporter,
    )

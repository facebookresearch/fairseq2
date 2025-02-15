# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from itertools import count
from typing import Generic, TypeVar, final

import torch
from torch.nn import Module
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.datasets import DataReader
from fairseq2.device import DeviceStatTracker
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.profilers import Profiler
from fairseq2.recipes.metrics import extend_batch_metrics
from fairseq2.recipes.utils.progress import NoopProgressReporter, ProgressReporter
from fairseq2.typing import CPU, DataType
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class GeneratorUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Generator`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None:
        """Process ``batch``."""

    @property
    @abstractmethod
    def model(self) -> Module:
        """The underlying model."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The generation-related metrics."""


BatchT = TypeVar("BatchT")


class AbstractGeneratorUnit(GeneratorUnit[BatchT]):
    """Provides a skeletal implementation of :class:`GeneratorUnit`."""

    def __init__(self, model: Module) -> None:
        self._model = model

    @final
    @property
    @override
    def model(self) -> Module:
        return self._model


@final
class Generator(Generic[BatchT]):
    """Generates output using a machine learning model."""

    _unit: GeneratorUnit[BatchT]
    _data_reader: DataReader[BatchT]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _metric_recorder: MetricRecorder
    _profiler: Profiler
    _device_stat_tracker: DeviceStatTracker
    _seed: int
    _wall_watch: Stopwatch
    _run: bool
    _progress_reporter: ProgressReporter

    def __init__(
        self,
        *,
        unit: GeneratorUnit[BatchT],
        data_reader: DataReader[BatchT],
        gangs: Gangs,
        dtype: DataType,
        amp: bool,
        metric_recorder: MetricRecorder,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        seed: int,
        wall_watch: Stopwatch,
    ) -> None:
        """
        :param unit: The generator unit.
        :param data_reader: The data reader.
        :param wall_watch: The stopwatch to track process wall-time.
        :param dtype: The data type of the model.
        :param amp: If ``True``, enables ``torch.amp``.
        :param seed: The random number generator seed.
        """
        self._unit = unit

        self._data_reader = data_reader

        if gangs.root.rank == 0:
            if gangs.dp.rank != 0 or gangs.tp.rank != 0:
                raise ValueError(
                    "The coordinator process of the root gang (i.e. rank 0) must be rank 0 in all parallel gangs."
                )

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._metric_recorder = metric_recorder

        self._profiler = profiler

        self._device_stat_tracker = device_stat_tracker

        self._seed = seed

        self._wall_watch = wall_watch

        self._run = False

        self._progress_reporter = NoopProgressReporter()

    @torch.inference_mode()
    def __call__(self, progress_reporter: ProgressReporter | None = None) -> None:
        if self._run:
            raise InvalidOperationError("The generator can only be run once.")

        self._run = True

        if progress_reporter is not None:
            self._progress_reporter = progress_reporter

        log.info("Running generation on {} device(s).", self._gangs.root.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Generation terminated!")

            raise

        self._gangs.close()

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Generation complete in {:,} seconds!", int(elapsed_time))

    def _do_run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._gangs.root.device)

        rng_bag.manual_seed(self._seed)

        watch = Stopwatch(start=True, device=self._gangs.root.device)

        self._unit.model.eval()

        num_effective_batches = 0

        with self._progress_reporter, self._profiler:
            task = self._progress_reporter.create_task("generate", total=None)

            self._device_stat_tracker.reset()

            for step_nr in count(start=1):
                with record_function(f"step_{step_nr}"):
                    task.step(1)

                    log.debug("Running step {}.", step_nr)

                    # Collect the batches.
                    with record_function(f"step_{step_nr}_data_load"):
                        try:
                            batches = next(self._data_reader)
                        except StopIteration:
                            break

                    # Call the unit.
                    for batch_nr, batch in enumerate(batches):
                        with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                            with self._maybe_autocast():
                                self._unit(batch)

                self._profiler.step()

                num_effective_batches += 1

            task.close()

        self._publish_metrics(num_effective_batches, watch.get_elapsed_time())

    def _maybe_autocast(self) -> AbstractContextManager[None]:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        return torch.autocast(
            device_type=self._gangs.root.device.type, dtype=self._dtype
        )

    def _publish_metrics(self, num_batches: int, elapsed_time: float) -> None:
        log.debug("Syncing metrics.")

        if self._gangs.tp.rank == 0:
            values = self._unit.metric_bag.sync_and_compute_metrics()
        else:
            values = None

        self._unit.metric_bag.reset_metrics()

        if self._gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            extend_batch_metrics(values, num_batches, elapsed_time)

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            values["elapsed_time"] = elapsed_time

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            self._metric_recorder.record_metrics("generate", values)

        self._gangs.root.barrier()

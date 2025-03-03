# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from itertools import count
from typing import Generic, TypeVar, final

import torch
from torch.profiler import record_function

from fairseq2.datasets import DataReader
from fairseq2.device import DeviceStatTracker
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.profilers import Profiler
from fairseq2.recipes.metrics import extend_batch_metrics
from fairseq2.recipes.model import Model
from fairseq2.recipes.utils.progress import NoopProgressReporter, ProgressReporter
from fairseq2.typing import CPU, ContextManager, DataType
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class EvalUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Evaluator` or :class:`Trainer`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None:
        """Process ``batch``."""

    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""
        pass

    @property
    def name(self) -> str | None:
        """The name of the unit for reporting purposes."""
        return None

    @property
    @abstractmethod
    def model(self) -> Model:
        """The underlying model."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The evaluation-related metrics."""


BatchT = TypeVar("BatchT")


@final
class Evaluator(Generic[BatchT]):
    """Evaluates a machine learning model."""

    _units: Sequence[EvalUnit[BatchT]]
    _data_readers: Sequence[DataReader[BatchT]]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _step_nr: int
    _metric_recorder: MetricRecorder
    _profiler: Profiler
    _device_stat_tracker: DeviceStatTracker
    _seed: int
    _wall_watch: Stopwatch
    _has_run: bool
    _progress_reporter: ProgressReporter

    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
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
        :param units: The evaluation units.
        :param data_readers: The data readers of ``units``.
        :param wall_watch: The stopwatch to track process wall-time.
        :param dtype: The data type of the model.
        :param amp: If ``True``, enables ``torch.amp``.
        :param seed: The random number generator seed.
        """
        if len(units) != len(data_readers):
            raise ValueError(
                f"The number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._units = units

        self._data_readers = data_readers

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._step_nr = 0

        self._metric_recorder = metric_recorder

        self._profiler = profiler

        self._device_stat_tracker = device_stat_tracker

        self._seed = seed

        self._wall_watch = wall_watch

        self._has_run = False

        self._progress_reporter = NoopProgressReporter()

    @torch.inference_mode()
    def __call__(self, progress_reporter: ProgressReporter | None = None) -> None:
        if self._has_run:
            raise InvalidOperationError("The evaluator can only be run once.")

        self._has_run = True

        if progress_reporter is not None:
            self._progress_reporter = progress_reporter

        rng_bag = RngBag.from_device_defaults(CPU, self._gangs.root.device)

        rng_bag.manual_seed(self._seed)

        log.info("Running evaluation on {} device(s).", self._gangs.root.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Evaluation terminated!")

            raise

        self._gangs.close()

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Evaluation complete in {:,} seconds!", int(elapsed_time))

    def _do_run(self) -> None:
        with self._progress_reporter, self._profiler:
            for unit, data_reader in zip(self._units, self._data_readers):
                if unit.name:
                    log.info("Evaluating {}.", unit.name)

                self._evaluate_unit(unit, data_reader)

    def _evaluate_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> None:
        watch = Stopwatch(start=True, device=self._gangs.root.device)

        unit.model.module.eval()

        num_effective_batches = 0

        task = self._progress_reporter.create_task("eval", total=None)

        self._device_stat_tracker.reset()

        for step_nr in count(start=1):
            self._step_nr += 1

            task.step(1)

            log.debug("Running step {}.", step_nr)

            # Collect the batches.
            with record_function(f"step_{self._step_nr}_data_load"):
                try:
                    batches = next(data_reader)
                except StopIteration:
                    break

                # Call the unit.
                for batch_nr, batch in enumerate(batches):
                    with record_function(f"step_{self._step_nr}_{batch_nr}_forward"):
                        with self._maybe_autocast():
                            unit(batch)

            self._profiler.step()

            num_effective_batches += 1

        task.close()

        self._publish_metrics(unit, num_effective_batches, watch.get_elapsed_time())

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        return torch.autocast(
            device_type=self._gangs.root.device.type, dtype=self._dtype
        )

    def _publish_metrics(
        self, unit: EvalUnit[BatchT], num_batches: int, elapsed_time: float
    ) -> None:
        log.debug("Syncing metrics.")

        if self._gangs.tp.rank == 0:
            values = unit.metric_bag.sync_and_compute_metrics()
        else:
            values = None

        unit.metric_bag.reset_metrics()

        if self._gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            extend_batch_metrics(values, num_batches, elapsed_time)

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            values["elapsed_time"] = elapsed_time

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            if unit.name:
                run_name = "eval/" + unit.name
            else:
                run_name = "eval"

            self._metric_recorder.record_metrics(run_name, values)

        self._gangs.root.barrier()

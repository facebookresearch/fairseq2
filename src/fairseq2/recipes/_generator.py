# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from itertools import count
from typing import Generic, TypeVar, final

import torch
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.datasets import DataReader, DataReadError
from fairseq2.device import DeviceStatTracker
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag, MetricBagError
from fairseq2.metrics.recorders import MetricRecorder, MetricRecordError
from fairseq2.profilers import Profiler
from fairseq2.recipes._error import RecipeError, UnitError
from fairseq2.recipes._metrics import extend_batch_metrics
from fairseq2.recipes._model import Model
from fairseq2.recipes._recipe import Recipe, RecipeStopException
from fairseq2.typing import CPU, ContextManager, DataType
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class GeneratorUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Generator`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None: ...

    @property
    @abstractmethod
    def model(self) -> Model: ...

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag: ...


BatchT = TypeVar("BatchT")


@final
class Generator(Recipe, Generic[BatchT]):
    """Generates output using a machine learning model."""

    _step_nr: int
    _unit: GeneratorUnit[BatchT]
    _data_reader: DataReader[BatchT]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _rng_bag: RngBag
    _seed: int
    _metric_recorder: MetricRecorder
    _profiler: Profiler
    _device_stat_tracker: DeviceStatTracker
    _data_watch: Stopwatch
    _compute_watch: Stopwatch
    _lapse_watch: Stopwatch
    _wall_watch: Stopwatch
    _progress_reporter: ProgressReporter
    _stop_requested: bool
    _num_batches_read: int
    _has_run: bool

    def __init__(
        self,
        *,
        unit: GeneratorUnit[BatchT],
        data_reader: DataReader[BatchT],
        gangs: Gangs,
        dtype: DataType,
        amp: bool,
        seed: int,
        metric_recorder: MetricRecorder,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        wall_watch: Stopwatch,
        progress_reporter: ProgressReporter,
    ) -> None:
        """
        :param unit: The generator unit.
        :param data_reader: The data reader.
        :param wall_watch: The stopwatch to track process wall-time.
        :param dtype: The data type of the model.
        :param amp: If ``True``, enables ``torch.amp``.
        :param seed: The random number generator seed.
        """
        self._step_nr = 0

        self._unit = unit

        self._data_reader = data_reader

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)

        self._seed = seed

        self._metric_recorder = metric_recorder

        self._profiler = profiler

        self._device_stat_tracker = device_stat_tracker

        self._data_watch = Stopwatch()

        self._compute_watch = Stopwatch(device=self._gangs.root.device)

        self._lapse_watch = Stopwatch()

        self._wall_watch = wall_watch

        self._progress_reporter = progress_reporter

        self._stop_requested = False

        self._num_batches_read = 0

        self._has_run = False

    @override
    @torch.inference_mode()
    def run(self) -> None:
        if self._has_run:
            raise InvalidOperationError("The generator has already been run.")

        self._has_run = True

        with self._progress_reporter:
            with self._rng_bag.temporary_manual_seed(self._seed):
                with self._profiler:
                    self._do_run()

        self._gangs.close()

    def _do_run(self) -> None:
        self._unit.model.module.eval()

        progress_task = self._progress_reporter.create_task("generate", total=None)

        self._device_stat_tracker.reset()

        with progress_task, self._lapse_watch:
            for step_nr in count(start=1):
                if self._stop_requested:
                    raise RecipeStopException()

                self._step_nr = step_nr

                progress_task.step(1)

                with record_function(f"step_{step_nr}"):
                    if not self._run_step():
                        break

                self._profiler.step()

        self._publish_metrics()

    def _run_step(self) -> bool:
        step_nr = self._step_nr

        log.debug("Running step {}.", step_nr)

        # Collect the batches.
        with self._data_watch:
            with record_function(f"step_{step_nr}_data_load"):
                try:
                    batches = next(self._data_reader)
                except DataReadError as ex:
                    raise RecipeError(
                        "The generator data read operation has failed. See the nested exception for details."
                    ) from ex
                except StopIteration:
                    return False

        # Call the unit.
        with self._compute_watch:
            for batch_nr, batch in enumerate(batches):
                with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                    with self._maybe_autocast():
                        try:
                            self._unit(batch)
                        except UnitError as ex:
                            raise RecipeError(
                                "The generator unit has failed. See the nested exception for details."
                            ) from ex

        self._num_batches_read += 1

        return True

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._dtype)

    def _publish_metrics(self) -> None:
        log.debug("Syncing generation metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = self._unit.metric_bag.sync_and_compute_metrics()
            else:
                values = None
        except MetricBagError as ex:
            raise RecipeError(
                "The generation metric values cannot be synced across processes. See the nested exception for details."
            ) from ex

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            data_time = self._data_watch.get_elapsed_time()

            compute_time = self._compute_watch.get_elapsed_time()

            extend_batch_metrics(
                values, self._num_batches_read, data_time + compute_time
            )

            values["data_time"] = data_time

            values["compute_time"] = compute_time

            values["lapse_time"] = self._lapse_watch.get_elapsed_time()

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            try:
                self._metric_recorder.record_metrics("generation", values)
            except MetricRecordError as ex:
                raise RecipeError(
                    "The generation metric values cannot recorded. See the nested exception for details."
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the metric sync operation has failed. See the nested exception for details."
            ) from ex

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr

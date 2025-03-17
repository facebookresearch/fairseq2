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


class EvalUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Evaluator` or :class:`Trainer`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None: ...

    def finalize(self) -> None:
        pass

    def set_train_step_nr(self, step_nr: int) -> None:
        pass

    @property
    def name(self) -> str | None:
        return None

    @property
    @abstractmethod
    def model(self) -> Model: ...

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag: ...


BatchT = TypeVar("BatchT")


@final
class Evaluator(Recipe, Generic[BatchT]):
    _step_nr: int
    _units: Sequence[EvalUnit[BatchT]]
    _data_readers: Sequence[DataReader[BatchT]]
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
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
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
        :param units: The evaluation units.
        :param data_readers: The data readers of ``units``.
        :param wall_watch: The stopwatch to track process wall-time.
        :param dtype: The data type of the model.
        :param amp: If ``True``, enables ``torch.amp``.
        :param seed: The random number generator seed.
        """
        self._step_nr = 0

        if len(units) == 0:
            raise ValueError("`units` must contain at least one evaluation unit.")

        if len(units) != len(data_readers):
            raise ValueError(
                f"The number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._units = units

        self._data_readers = data_readers

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
            raise InvalidOperationError("The evaluator has already been run.")

        self._has_run = True

        with self._progress_reporter:
            with self._rng_bag.temporary_manual_seed(self._seed):
                with self._profiler:
                    self._do_run()

        self._gangs.close()

    def _do_run(self) -> None:
        for unit, data_reader in zip(self._units, self._data_readers):
            if unit.name:
                log.info("Evaluating {}.", unit.name)

            self._run_unit(unit, data_reader)

    def _run_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> None:
        unit.model.module.eval()

        progress_task = self._progress_reporter.create_task("eval", total=None)

        self._device_stat_tracker.reset()

        with progress_task, self._lapse_watch:
            for step_nr in count(start=self._step_nr + 1):
                if self._stop_requested:
                    raise RecipeStopException()

                self._step_nr = step_nr

                progress_task.step(1)

                with record_function(f"step_{step_nr}"):
                    if not self._run_step(unit, data_reader):
                        break

                self._profiler.step()

            with self._compute_watch:
                with record_function("finalize"):
                    with self._maybe_autocast():
                        try:
                            unit.finalize()
                        except UnitError as ex:
                            s = "evaluator"

                            if unit.name:
                                s = f"'{unit.name}' {s}"

                            raise RecipeError(
                                f"The {s} unit has failed to finalize. See the nested exception for details."
                            ) from ex

        self._publish_metrics(unit)

        self._reset_watches()

        self._num_batches_read = 0

    def _run_step(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> bool:
        step_nr = self._step_nr

        log.debug("Running step {}.", step_nr)

        # Collect the batches.
        with self._data_watch:
            with record_function(f"step_{step_nr}_data_load"):
                try:
                    batches = next(data_reader)
                except DataReadError as ex:
                    s = "evaluator"

                    if unit.name:
                        s = f"'{unit.name}' {s}"

                    raise RecipeError(
                        f"The {s} data read operation has failed. See the nested exception for details."
                    ) from ex
                except StopIteration:
                    return False

        # Call the unit.
        with self._compute_watch:
            for batch_nr, batch in enumerate(batches):
                with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                    with self._maybe_autocast():
                        try:
                            unit(batch)
                        except UnitError as ex:
                            s = "evaluator"

                            if unit.name:
                                s = f"'{unit.name}' {s}"

                            raise RecipeError(
                                f"The {s} unit has failed. See the nested exception for details."
                            ) from ex

        self._num_batches_read += 1

        return True

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._dtype)

    def _publish_metrics(self, unit: EvalUnit[BatchT]) -> None:
        log.debug("Syncing evaluation metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = unit.metric_bag.sync_and_compute_metrics()
            else:
                values = None
        except MetricBagError as ex:
            s = "evaluation"

            if unit.name:
                s = f"'{unit.name}' {s}"

            raise RecipeError(
                f"The {s} metric values cannot be synced across processes. See the nested exception for details."
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

            run = "eval"

            if unit.name:
                run = f"{run}/{unit.name}"

            try:
                self._metric_recorder.record_metrics(run, values)
            except MetricRecordError as ex:
                s = "evaluation"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} metric values cannot recorded. See the nested exception for details."
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the metric sync operation has failed. See the nested exception for details."
            ) from ex

    def _reset_watches(self) -> None:
        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr

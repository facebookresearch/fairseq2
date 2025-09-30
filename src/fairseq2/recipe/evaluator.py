# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping, Sequence
from contextlib import nullcontext
from typing import Any, Generic, TypeVar, final

import torch
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.datasets import DataReader
from fairseq2.device import CPU, SupportsDeviceTransfer
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.metrics import MetricBag, sync_and_compute_metrics
from fairseq2.metrics.common import extend_batch_metric_values
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.profilers import Profiler
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.task import Task, TaskStopException
from fairseq2.typing import ContextManager
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


class EvalUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Evaluator` or :class:`Trainer`."""

    @abstractmethod
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None: ...

    def set_train_step_nr(self, step_nr: int) -> None:
        pass

    @abstractmethod
    def process_batch(self, batch: BatchT_contra, metric_bag: MetricBag) -> None: ...

    def finalize(self, metric_bag: MetricBag) -> None:
        pass

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        pass

    @property
    def name(self) -> str | None:
        return None

    @property
    @abstractmethod
    def model(self) -> RecipeModel: ...


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Evaluator(Task):
    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
        gangs: Gangs,
        amp: bool,
        amp_dtype: DataType,
        metric_recorder: MetricRecorder,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        wall_watch: Stopwatch,
        progress_reporter: ProgressReporter,
        seed: int,
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
                f"Number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._units = units

        self._data_readers = data_readers

        self._gangs = gangs

        self._amp = amp

        self._amp_dtype = amp_dtype

        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)

        self._metric_recorder = metric_recorder

        self._profiler = profiler

        self._device_stat_tracker = device_stat_tracker

        self._data_watch = Stopwatch()

        self._compute_watch = Stopwatch(device=self._gangs.root.device)

        self._lapse_watch = Stopwatch()

        self._wall_watch = wall_watch

        self._progress_reporter = progress_reporter

        self._seed = seed

        self._stop_requested = False

        self._num_batches_read = 0

        self._has_run = False

    @override
    @torch.inference_mode()
    def run(self) -> None:
        if self._has_run:
            raise InvalidOperationError("Evaluator has already been run.")

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

    def _run_unit(self, unit: EvalUnit[Any], data_reader: DataReader[Any]) -> None:
        unit.model.module.eval()

        metric_bag = MetricBag(device=self._gangs.root.device)

        unit.prepare_metric_bag(metric_bag)

        progress_task = self._progress_reporter.create_task("eval", total=None)

        self._device_stat_tracker.reset()

        eod = False

        with progress_task, self._lapse_watch:
            while not eod:
                if self._stop_requested:
                    raise TaskStopException()

                batches = self._read_next_batches(unit, data_reader)
                if batches is None:
                    eod = True

                    continue

                self._step_nr += 1

                progress_task.step()

                with record_function(f"step_{self._step_nr}"):
                    self._run_step(unit, batches, metric_bag)

                self._profiler.step()

            with self._compute_watch:
                with record_function("finalize"):
                    self._call_unit_finalize(unit, metric_bag)

        self._publish_metrics(unit, metric_bag)

    def _read_next_batches(
        self, unit: EvalUnit[Any], data_reader: DataReader[Any]
    ) -> list[Any] | None:
        with self._data_watch:
            try:
                batches = next(data_reader)
            except StopIteration:
                batches = None

            if batches is None:
                data_reader.reset()

        return batches

    def _run_step(
        self, unit: EvalUnit[Any], batches: list[Any], metric_bag: MetricBag
    ) -> None:
        log.debug("Running step {}.", self._step_nr)

        with self._compute_watch:
            batches.reverse()

            num_batches = len(batches)

            for batch_nr in range(num_batches):
                batch = batches.pop()

                batch.to(self._gangs.root.device, non_blocking=True)

                with record_function(f"step_{self._step_nr}_{batch_nr}"):
                    self._call_unit(unit, batch, metric_bag)

        self._num_batches_read += 1

    def _call_unit(
        self, unit: EvalUnit[Any], batch: Any, metric_bag: MetricBag
    ) -> None:
        with self._maybe_autocast():
            unit.process_batch(batch, metric_bag)

    def _call_unit_finalize(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        with self._maybe_autocast():
            unit.finalize(metric_bag)

    def _maybe_autocast(self) -> ContextManager[None]:
        if not self._amp or self._amp_dtype == torch.float32:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _publish_metrics(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        log.debug("Syncing evaluation metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = sync_and_compute_metrics(metric_bag, gangs.dp)
            else:
                values = None
        except GangError as ex:
            raise_operational_gang_error(ex)

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            values = {k: v for k, v in values.items() if not k.startswith("total_")}

            unit.process_metric_values(values)

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            data_time = self._data_watch.get_elapsed_time()

            compute_time = self._compute_watch.get_elapsed_time()

            extend_batch_metric_values(
                values, self._num_batches_read, data_time + compute_time
            )

            values["data_time"] = data_time

            values["compute_time"] = compute_time

            values["lapse_time"] = self._lapse_watch.get_elapsed_time()

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            category = "eval"

            if unit.name:
                category = f"{category}/{unit.name}"

            self._metric_recorder.record_metric_values(category, values)

        gangs.root.barrier()

        self._reset_lapse_state()

    def _reset_lapse_state(self) -> None:
        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

        self._device_stat_tracker.reset()

        self._num_batches_read = 0

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

    @override
    def close(self) -> None:
        self._metric_recorder.close()

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr

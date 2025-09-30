# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
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


class GeneratorUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Generator`."""

    @abstractmethod
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None: ...

    @abstractmethod
    def process_batch(self, batch: BatchT_contra, metric_bag: MetricBag) -> None: ...

    def finalize(self, metric_bag: MetricBag) -> None:
        pass

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        pass

    @property
    @abstractmethod
    def model(self) -> RecipeModel: ...


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Generator(Task):
    """Generates output using a machine learning model."""

    def __init__(
        self,
        *,
        unit: GeneratorUnit[BatchT],
        data_reader: DataReader[BatchT],
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

        self._amp = amp

        self._amp_dtype = amp_dtype

        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)

        self._metric_bag = MetricBag(device=gangs.root.device)

        unit.prepare_metric_bag(self._metric_bag)

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
            raise InvalidOperationError("Generator has already been run.")

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

        eod = False

        with progress_task, self._lapse_watch:
            while not eod:
                if self._stop_requested:
                    raise TaskStopException()

                batches = self._read_next_batches()
                if batches is None:
                    eod = True

                    continue

                self._step_nr += 1

                progress_task.step()

                with record_function(f"step_{self._step_nr}"):
                    self._run_step(batches)

                self._profiler.step()

            with self._compute_watch:
                with record_function("finalize"):
                    self._call_unit_finalize()

        self._publish_metrics()

    def _read_next_batches(self) -> list[Any] | None:
        with self._data_watch:
            try:
                batches = next(self._data_reader)
            except StopIteration:
                batches = None

            if batches is None:
                self._data_reader.reset()

        return batches

    def _run_step(self, batches: list[Any]) -> None:
        log.debug("Running step {}.", self._step_nr)

        with self._compute_watch:
            batches.reverse()

            num_batches = len(batches)

            for batch_nr in range(num_batches):
                batch = batches.pop()

                batch.to(self._gangs.root.device, non_blocking=True)

                with record_function(f"step_{self._step_nr}_{batch_nr}"):
                    self._call_unit(batch)

        self._num_batches_read += 1

    def _call_unit(self, batch: Any) -> None:
        with self._maybe_autocast():
            self._unit.process_batch(batch, self._metric_bag)

    def _call_unit_finalize(self) -> None:
        with self._maybe_autocast():
            self._unit.finalize(self._metric_bag)

    def _maybe_autocast(self) -> ContextManager[None]:
        if not self._amp or self._amp_dtype == torch.float32:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _publish_metrics(self) -> None:
        log.debug("Syncing generation metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = sync_and_compute_metrics(self._metric_bag, gangs.dp)
            else:
                values = None
        except GangError as ex:
            raise_operational_gang_error(ex)

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            values = {k: v for k, v in values.items() if not k.startswith("total_")}

            self._unit.process_metric_values(values)

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

            self._metric_recorder.record_metric_values("generation", values)

        gangs.root.barrier()

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

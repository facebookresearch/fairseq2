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
from torch.nn import Module
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.datasets import DataReader, DataReadError
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag, sync_and_compute_metrics
from fairseq2.metrics.common import extend_batch_metric_values
from fairseq2.metrics.recorders import MetricRecorder, MetricRecordError
from fairseq2.profilers import Profiler
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.closable import Closable
from fairseq2.typing import ContextManager
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.warn import _warn_deprecated

BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


class EvalUnit(ABC, Generic[BatchT_contra]):
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
    def name(self) -> str:
        return "default"

    @property
    def model(self) -> RecipeModel:
        _warn_deprecated("`EvalUnit.model` is deprecated and will be removed in v0.14.")

        raise NotImplementedError()


class EvaluatorError(Exception):
    pass


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Evaluator(Closable):
    """Evaluates a machine learning model."""

    def __init__(
        self,
        *,
        model: Module,
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
        rng_bag: RngBag,
        seed: int,
    ) -> None:
        if len(units) == 0:
            raise ValueError("`units` must contain at least one evaluation unit.")

        if len(units) != len(data_readers):
            raise ValueError(
                f"Number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._model = model
        self._units = units
        self._data_readers = data_readers
        self._gangs = gangs
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._metric_recorder = metric_recorder
        self._profiler = profiler
        self._device_stat_tracker = device_stat_tracker
        self._data_watch = Stopwatch()
        self._compute_watch = Stopwatch(device=gangs.device)
        self._lapse_watch = Stopwatch()
        self._wall_watch = wall_watch
        self._progress_reporter = progress_reporter
        self._rng_bag = rng_bag
        self._seed = seed
        self._step_nr = 0
        self._stop_requested = False
        self._num_batches_read = 0
        self._has_run = False

    @torch.inference_mode()
    def run(self) -> bool:
        """
        :raises EvaluatorError:
        :raises InvalidOperationError:
        """
        if self._has_run:
            raise InvalidOperationError("Evaluator has already been run.")

        self._has_run = True

        with self._progress_reporter:
            with self._rng_bag.temporary_manual_seed(self._seed):
                with self._profiler:
                    done = self._do_run()

        self._gangs.close()

        return done

    def _do_run(self) -> bool:
        self._model.eval()

        for unit, data_reader in zip(self._units, self._data_readers):
            if self._units:
                log.info("Evaluating '{}' unit.", unit.name)

            done = self._run_unit(unit, data_reader)
            if not done:
                return False

        return True

    def _run_unit(self, unit: EvalUnit[Any], data_reader: DataReader[Any]) -> bool:
        metric_bag = MetricBag(device=self._gangs.device)

        unit.prepare_metric_bag(metric_bag)

        progress_task = self._progress_reporter.create_task("eval", total=None)

        self._device_stat_tracker.reset()

        eod = False

        with progress_task, self._lapse_watch:
            while not eod:
                if self._stop_requested:
                    return False

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

        return True

    def _read_next_batches(
        self, unit: EvalUnit[Any], data_reader: DataReader[Any]
    ) -> list[Any] | None:
        with self._data_watch:
            try:
                batches = next(data_reader)
            except DataReadError as ex:
                raise EvaluatorError(
                    f"Failed to read data at step {self._step_nr} for '{unit.name}' unit."
                ) from ex
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

                batch.to(self._gangs.device, non_blocking=True)

                with record_function(f"step_{self._step_nr}_{batch_nr}"):
                    self._call_unit(unit, batch, metric_bag)

        self._num_batches_read += 1

    def _call_unit(
        self, unit: EvalUnit[Any], batch: Any, metric_bag: MetricBag
    ) -> None:
        try:
            with self._maybe_autocast():
                unit.process_batch(batch, metric_bag)
        except (RuntimeError, OSError, GangError, DataReadError) as ex:
            raise EvaluatorError(
                f"'{unit.name}' unit failed at step {self._step_nr}."
            ) from ex

    def _call_unit_finalize(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        try:
            with self._maybe_autocast():
                unit.finalize(metric_bag)
        except (RuntimeError, OSError, GangError) as ex:
            raise EvaluatorError(f"'{unit.name}' unit failed to finalize.") from ex

    def _maybe_autocast(self) -> ContextManager[None]:
        if not self._amp or self._amp_dtype == torch.float32:
            return nullcontext()

        return torch.autocast(
            device_type=self._gangs.device.type, dtype=self._amp_dtype
        )

    def _publish_metrics(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        try:
            self._do_publish_metrics(unit, metric_bag)
        except (GangError, MetricRecordError) as ex:
            raise EvaluatorError(
                f"Failed to publish metrics of '{unit.name}' unit."
            ) from ex

    def _do_publish_metrics(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        log.debug("Syncing evaluation metrics.")

        gangs = self._gangs

        if gangs.tp.rank == 0:
            values = sync_and_compute_metrics(metric_bag, gangs.dp)
        else:
            values = None

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

            if self._units:
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

    def request_stop(self) -> None:
        self._stop_requested = True

    @override
    def close(self) -> None:
        self._metric_recorder.close()

    @property
    def step_nr(self) -> int:
        return self._step_nr

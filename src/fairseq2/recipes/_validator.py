# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, TypeVar, final

import torch
from torch import Tensor
from torch.profiler import record_function

from fairseq2.checkpoint import CheckpointError, CheckpointManager, CheckpointSaveError
from fairseq2.data_type import DataType
from fairseq2.datasets import DataReader, DataReadError
from fairseq2.device import CPU, SupportsDeviceTransfer
from fairseq2.error import ContractError, InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.metrics import (
    MetricBag,
    MetricBagError,
    sync_and_compute_metrics,
)
from fairseq2.metrics.recorders import (
    MetricDescriptor,
    MetricRecorder,
    MetricRecordError,
)
from fairseq2.profilers import Profiler
from fairseq2.recipes.metrics import extend_batch_metric_values
from fairseq2.typing import ContextManager
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

# isort: split

from fairseq2.recipes._error import RecipeError, UnitError
from fairseq2.recipes._evaluator import EvalUnit

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Validator:
    _step_nr: int
    _units: Sequence[EvalUnit[Any]]
    _data_readers: Sequence[DataReader[Any]]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _rng_bag: RngBag
    _seed: int
    _score_metric_descriptor: MetricDescriptor | None
    _checkpoint_manager: CheckpointManager
    _metric_recorder: MetricRecorder
    _profiler: Profiler
    _device_stat_tracker: DeviceStatTracker
    _data_watch: Stopwatch
    _compute_watch: Stopwatch
    _lapse_watch: Stopwatch
    _wall_watch: Stopwatch
    _progress_reporter: ProgressReporter
    _num_batches_read: int
    _has_run: bool
    _best_score_and_step: tuple[float, int]

    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
        gangs: Gangs,
        dtype: DataType,
        amp: bool,
        score_metric_descriptor: MetricDescriptor | None,
        checkpoint_manager: CheckpointManager,
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
        :param score_metric_descriptor: The descriptor of the metric to use for
            score calculation.
        :param seed: The random number generator seed.
        """
        self._step_nr = 0

        if len(units) == 0:
            raise ValueError("`units` must contain at least one validation unit.")

        if len(units) != len(data_readers):
            raise ValueError(
                f"The number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._units = units

        self._data_readers = data_readers

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._score_metric_descriptor = score_metric_descriptor

        self._checkpoint_manager = checkpoint_manager

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

        self._num_batches_read = 0

        self._has_run = False

        self._best_score = -torch.inf

        self._best_step_nr = -1

    @torch.inference_mode()
    def run(self, train_step_nr: int) -> float | None:
        if self._has_run:
            raise InvalidOperationError("The validator has already been run.")

        self._has_run = True

        self._maybe_restore_best_score_and_step()

        with self._rng_bag.temporary_manual_seed(self._seed):
            with self._profiler:
                return self._do_run(train_step_nr)

    def _maybe_restore_best_score_and_step(self) -> None:
        if self._score_metric_descriptor is None:
            return

        if self._best_step_nr >= 0:
            return

        try:
            if self._gangs.root.rank == 0:
                scores_and_steps = self._checkpoint_manager.load_scores()
            else:
                scores_and_steps = []
        except CheckpointError as ex:
            raise RecipeError(
                "The past training scores cannot be loaded. See the nested exception for details."
            ) from ex

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the training score load operation has failed. See the nested exception for details."
            ) from ex

        if scores_and_steps:
            self._best_score, self._best_step_nr = scores_and_steps[0]
        else:
            self._best_step_nr = 0

    def _do_run(self, train_step_nr: int) -> float | None:
        scores = []

        for unit, data_reader in zip(self._units, self._data_readers):
            if unit.name:
                log.info("Validating {}.", unit.name)

            score = self._run_unit(train_step_nr, unit, data_reader)

            if score is not None:
                scores.append(score)

        score = self._compute_aggregated_score(scores)

        self._update_best_score(train_step_nr, score)

        return score

    def _run_unit(
        self, train_step_nr: int, unit: EvalUnit[Any], data_reader: DataReader[Any]
    ) -> float | None:
        unit.model.module.eval()

        metric_bag = MetricBag(device=self._gangs.root.device)

        progress_task = self._progress_reporter.create_task("valid", total=None)

        self._device_stat_tracker.reset()

        eod = False

        with progress_task, self._lapse_watch:
            try:
                unit.set_train_step_nr(train_step_nr)
            except UnitError as ex:
                s = "validator"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} unit has failed. See the nested exception for details."
                ) from ex

            while not eod:
                try:
                    self._checkpoint_manager.maybe_complete_async_checkpoint()
                except CheckpointSaveError as ex:
                    raise RecipeError(
                        f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
                    ) from ex

                batches = self._read_next_batches(unit, data_reader)
                if batches is None:
                    eod = True

                    continue

                self._step_nr += 1

                progress_task.step(1)

                with record_function(f"step_{self._step_nr}"):
                    self._run_step(unit, batches, metric_bag)

                self._profiler.step()

            with self._compute_watch:
                with record_function("finalize"):
                    self._call_unit_finalize(unit, metric_bag)

        metric_values = self._publish_metrics(train_step_nr, unit, metric_bag)

        return self._maybe_get_unit_score(metric_values)

    def _read_next_batches(
        self, unit: EvalUnit[Any], data_reader: DataReader[Any]
    ) -> list[Any] | None:
        with self._data_watch:
            try:
                batches = next(data_reader)
            except DataReadError as ex:
                s = "validator"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} data read operation has failed. See the nested exception for details."
                ) from ex
            except StopIteration:
                batches = None

            if batches is None:
                data_reader.reset()

        return batches

    def _run_step(
        self, unit: EvalUnit[Any], batches: list[Any], metric_bag: MetricBag
    ) -> None:
        log.debug("Running validation step {}.", self._step_nr)

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
            try:
                unit(batch, metric_bag)
            except UnitError as ex:
                s = "validator"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} unit has failed. See the nested exception for details."
                ) from ex

    def _call_unit_finalize(self, unit: EvalUnit[Any], metric_bag: MetricBag) -> None:
        with self._maybe_autocast():
            try:
                unit.finalize(metric_bag)
            except UnitError as ex:
                s = "validator"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} unit has failed to finalize. See the nested exception for details."
                ) from ex

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._dtype)

    def _publish_metrics(
        self, train_step_nr: int, unit: EvalUnit[Any], metric_bag: MetricBag
    ) -> dict[str, object]:
        log.debug("Syncing validation metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = sync_and_compute_metrics(metric_bag, gangs.dp)
            else:
                values = None
        except MetricBagError as ex:
            s = "validation"

            if unit.name:
                s = f"'{unit.name}' {s}"

            raise RecipeError(
                f"The {s} metric values cannot be synced across processes. See the nested exception for details."
            ) from ex

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

            section = "valid"

            if unit.name is not None:
                section = f"{section}/{unit.name}"

            try:
                self._metric_recorder.record_metric_values(
                    section, values, train_step_nr
                )
            except MetricRecordError as ex:
                s = "validation"

                if unit.name:
                    s = f"'{unit.name}' {s}"

                raise RecipeError(
                    f"The {s} metric values of step {train_step_nr} cannot recorded. See the nested exception for details."
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the metric sync operation has failed. See the nested exception for details."
            ) from ex

        self._reset_lapse_state()

        if values is None:
            values = {}

        return values

    def _reset_lapse_state(self) -> None:
        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

        self._device_stat_tracker.reset()

        self._num_batches_read = 0

    def _maybe_get_unit_score(self, metric_values: dict[str, object]) -> float | None:
        if self._score_metric_descriptor is None:
            return None

        if self._gangs.root.rank != 0:
            return None

        metric_value = metric_values.get(self._score_metric_descriptor.name)
        if metric_value is None:
            return None

        if not isinstance(metric_value, (int, float, Tensor)):
            raise ContractError(
                f"The score metric value must be of type `int`, `float`, or `{Tensor}`, but is of type `{type(metric_value)}` instead."
            )

        score = float(metric_value)

        if not self._score_metric_descriptor.higher_better:
            score = -score

        return score

    def _compute_aggregated_score(self, scores: list[float]) -> float | None:
        if self._score_metric_descriptor is None:
            return None

        if self._gangs.root.rank != 0:
            return 0.0

        if not scores:
            raise ContractError(
                "None of the evaluation units returned a score metric value."
            )

        return sum(scores)

    def _update_best_score(self, train_step_nr: int, score: float | None) -> None:
        if self._score_metric_descriptor is None:
            return

        if score is None:
            raise InternalError("`score` is `None`.")

        if self._gangs.root.rank != 0:
            return

        if score > self._best_score:
            self._best_score, self._best_step_nr = score, train_step_nr

        if not log.is_enabled_for_info():
            return

        best_score = self._best_score

        metric_descriptor = self._score_metric_descriptor

        if not metric_descriptor.higher_better:
            score, best_score = -score, -best_score

        v1 = metric_descriptor.formatter(score)
        v2 = metric_descriptor.formatter(best_score)

        log.info("Score - Metric: {} | Last: {} (step {}) | Best: {} (step {})", metric_descriptor.display_name, v1, train_step_nr, v2, self._best_step_nr)  # fmt: skip

    def reset(self) -> None:
        self._step_nr = 0

        for data_reader in self._data_readers:
            data_reader.reset()

        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

        self._num_batches_read = 0

        self._has_run = False

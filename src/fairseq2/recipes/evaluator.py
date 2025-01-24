# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from itertools import count
from typing import Generic, TypeVar, final

import torch
from torch.nn import Module
from typing_extensions import override

from fairseq2.datasets import DataReader
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.recorders import MetricRecorder, record_metrics
from fairseq2.recipes.metrics import extend_batch_metrics
from fairseq2.recipes.utils.rich import create_rich_progress
from fairseq2.typing import CPU, DataType
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import RngBag

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class EvalUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Evaluator` or :class:`Trainer`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None:
        """Process ``batch``."""

    @abstractmethod
    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""

    @property
    @abstractmethod
    def model(self) -> Module:
        """The underlying model."""

    @property
    @abstractmethod
    def display_name(self) -> str | None:
        """The display name of the unit for reporting purposes."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The evaluation-related metrics."""


BatchT = TypeVar("BatchT")


class AbstractEvalUnit(EvalUnit[BatchT]):
    """Provides a skeletal implementation of :class:`EvalUnit`."""

    _model: Module
    _display_name: str | None

    def __init__(self, model: Module, *, display_name: str | None = None) -> None:
        self._model = model
        self._display_name = display_name

    @override
    def set_step_nr(self, step_nr: int) -> None:
        pass

    @final
    @property
    @override
    def model(self) -> Module:
        return self._model

    @final
    @property
    @override
    def display_name(self) -> str | None:
        return self._display_name


@final
class Evaluator(Generic[BatchT]):
    """Evaluates a machine learning model."""

    _units: Sequence[EvalUnit[BatchT]]
    _data_readers: Sequence[DataReader[BatchT]]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _metric_recorders: Sequence[MetricRecorder]
    _seed: int
    _wall_watch: Stopwatch
    _run: bool

    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
        gangs: Gangs,
        metric_recorders: Sequence[MetricRecorder],
        wall_watch: Stopwatch,
        dtype: DataType = torch.float32,
        amp: bool = False,
        seed: int = 2,
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

        if gangs.root.rank == 0:
            if gangs.dp.rank != 0 or gangs.tp.rank != 0:
                raise ValueError(
                    "The coordinator process of the root gang (i.e. rank 0) must be rank 0 in all parallel gangs."
                )

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._metric_recorders = metric_recorders

        self._seed = seed

        self._wall_watch = wall_watch

        self._run = False

    @torch.inference_mode()
    def __call__(self) -> None:
        if self._run:
            raise InvalidOperationError("The evaluator can only be run once.")

        self._run = True

        log.info("Running evaluation on {} device(s).", self._gangs.root.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Evaluation terminated!")

            raise

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Evaluation complete in {:,} seconds!", int(elapsed_time))

    def _do_run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._gangs.root.device)

        rng_bag.manual_seed(self._seed)

        for unit, data_reader in zip(self._units, self._data_readers):
            if unit.display_name:
                log.info("Evaluating {}.", unit.display_name)

            self._evaluate_unit(unit, data_reader)

    def _evaluate_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> None:
        watch = Stopwatch(start=True, device=self._gangs.root.device)

        unit.model.eval()

        num_effective_batches = 0

        with create_rich_progress() as progress:
            task = progress.add_task("eval", total=None)

            for step_nr in count(start=1):
                progress.update(task, advance=1)

                log.debug("Running step {}.", step_nr)

                try:
                    batches = next(data_reader)
                except StopIteration:
                    break

                for batch in batches:
                    with self._maybe_autocast():
                        unit(batch)

                num_effective_batches += 1

        self._publish_metrics(unit, num_effective_batches, watch.get_elapsed_time())

    def _maybe_autocast(self) -> AbstractContextManager[None]:
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

        if self._gangs.root.rank != 0:
            return

        if values is None:
            raise InternalError(
                "The synchronized metric values are `None`. Please file a bug report."
            )

        extend_batch_metrics(values, num_batches, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        if unit.display_name:
            run_name = "eval/" + unit.display_name
        else:
            run_name = "eval"

        record_metrics(self._metric_recorders, run_name, values)

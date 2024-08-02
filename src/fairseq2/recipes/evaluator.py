# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, final

import torch
from torch.nn import Module
from typing_extensions import override

from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang, all_sum
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    JsonFileMetricRecorder,
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.common_metrics import set_throughput_value
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import CPU
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT")

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
    def display_name(self) -> Optional[str]:
        """The display name of the unit for reporting purposes."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The evaluation-related metrics."""


class AbstractEvalUnit(EvalUnit[BatchT]):
    """Provides a skeletal implementation of :class:`EvalUnit`."""

    _model: Module
    _display_name: Optional[str]

    def __init__(self, model: Module, *, display_name: Optional[str] = None) -> None:
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
    def display_name(self) -> Optional[str]:
        return self._display_name


@final
class Evaluator(Generic[BatchT]):
    """Evaluates a machine learning model."""

    _units: Sequence[EvalUnit[BatchT]]
    _data_readers: Sequence[DataReader[BatchT]]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _metric_recorders: list[MetricRecorder]
    _seed: int
    _wall_watch: Stopwatch
    _run: bool

    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
        root_gang: Gang,
        wall_watch: Stopwatch,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        tb_dir: Optional[Path] = None,
        metrics_dir: Optional[Path] = None,
        seed: int = 2,
    ) -> None:
        """
        :param units:
            The evaluation units.
        :param data_readers:
            The data readers corresponding to each unit in ``units``.
        :param root_gang:
            The gang for distributed evaluation.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dp_gang:
            The data parallel gang. If ``None``, ``root_gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models.
        :param tb_dir:
            The TensorBoard log directory to dump metrics.
        :param metrics_dir:
            The directory to dump metrics.
        :param seed:
            The random number generator seed.
        """
        if len(units) != len(data_readers):
            raise ValueError(
                f"The number of data readers in `data_readers` must match the number of units in `units` ({len(units)}), but is {len(data_readers)} instead."
            )

        self._units = units

        self._data_readers = data_readers

        self._root_gang = root_gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = root_gang
            self._tp_gang = FakeGang(device=root_gang.device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        if root_gang.rank == 0:
            if self._dp_gang.rank != 0 or self._tp_gang.rank != 0:
                raise ValueError(
                    f"The coordinator process of `root_gang` (i.e. rank 0) must be rank 0 in `dp_gang` and `tp_gang`, but is {self._dp_gang.rank} and {self._tp_gang.rank} instead."
                )

        if root_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if tb_dir is not None:
                self._metric_recorders.append(TensorBoardRecorder(tb_dir))

            if metrics_dir is not None:
                self._metric_recorders.append(JsonFileMetricRecorder(metrics_dir))
        else:
            self._metric_recorders = []

        self._seed = seed

        self._wall_watch = wall_watch

        self._run = False

    @torch.inference_mode()
    def __call__(self) -> None:
        if self._run:
            raise RuntimeError("The evaluator can only be run once.")

        self._run = True

        log.info("Running evaluation on {} device(s).", self._root_gang.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Evaluation terminated!")

            raise

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Evaluation complete in {:,} seconds!", int(elapsed_time))

    def _do_run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._root_gang.device)

        rng_bag.manual_seed(self._seed)

        for unit, data_reader in zip(self._units, self._data_readers):
            if unit.display_name:
                log.info("Evaluating {}.", unit.display_name)

            self._evaluate_unit(unit, data_reader)

    def _evaluate_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> None:
        watch = Stopwatch(start=True, device=self._root_gang.device)

        unit.model.eval()

        with create_rich_progress() as progress:
            task = progress.add_task("eval", total=None)

            for step_nr in count(start=1):
                progress.update(task, advance=1)

                log.debug("Running step {}.", step_nr)

                try:
                    batches = next(data_reader)
                except StopIteration:
                    batches = []

                for batch in batches:
                    unit(batch)

                if self._is_eod(batches):
                    break

        self._publish_metrics(unit, watch.get_elapsed_time())

    def _is_eod(self, batches: list[BatchT]) -> bool:
        total_num_batches = all_sum(self._dp_gang, len(batches))

        return bool(total_num_batches == 0)

    def _publish_metrics(self, unit: EvalUnit[BatchT], elapsed_time: float) -> None:
        log.debug("Syncing metrics.")

        if self._tp_gang.rank == 0:
            values = unit.metric_bag.sync_and_compute_metrics()
        else:
            values = None

        unit.metric_bag.reset_metrics()

        if self._root_gang.rank != 0:
            return

        assert values is not None

        set_throughput_value(values, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        if unit.display_name:
            run_name = "eval/" + unit.display_name
        else:
            run_name = "eval"

        record_metrics(self._metric_recorders, run_name, values)


@final
class HFEvaluator(Generic[BatchT]):
    """Evaluate a machine learning model with HuggingFace's evaluate.Metric library"""

    _model: Model
    _preprocessor: Callable[[BatchT], Tuple[SequenceBatch, SequenceBatch]]
    _postprocessor: Callable[[Any, SequenceBatch], Tuple[List[str], List[str]]]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _data_reader: DataReader[BatchT]
    _metric_recorders: List[MetricRecorder]
    _wall_watch: Stopwatch
    _elapsed_time: float
    _run: bool

    def __init__(
        self,
        model: Model,
        metrics: List[str],
        gang: Gang,
        data_reader: DataReader[BatchT],
        wall_watch: Stopwatch,
        preprocessor: Callable[[BatchT], Tuple[SequenceBatch, SequenceBatch]],
        postprocessor: Callable[[Any, SequenceBatch], Tuple[List[str], List[str]]],
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        tb_dir: Optional[Path] = None,
    ) -> None:
        """
        :param model:
            The fairseq2 machine learning model to be evaluate
        :param metrics:
            The list of metric names implemented in HuggingFace.evaluate
        :param gang:
            The gang to use for distributed evaluation.
        :param data_reader:
            The data reader of the eval split.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param preprocessor:
            The preprocessor to convert the batch into inputs and targets SequenceBatch objects.
        :param postprocessor:
            The postprocessor to convert the model outputs and target sequences into predictions and references.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models.
        :param tb_dir:
            The TensorBoard log directory to dump metrics.
        """
        try:
            evaluate = importlib.import_module("evaluate")
        except ImportError as exc:
            raise ImportError(
                "HFMetric requires the library `evaluate`, for instance via `pip install evaluate`"
            ) from exc

        self._model = model

        self._root_gang = gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = gang
            self._tp_gang = FakeGang(device=gang.device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        self._data_reader = data_reader

        self._metrics = evaluate.combine(metrics)

        self._preprocessor = preprocessor

        self._postprocessor = postprocessor

        if self._tp_gang.rank == 0 and self._dp_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if tb_dir is not None:
                self._metric_recorders.append(TensorBoardRecorder(tb_dir))
        else:
            self._metric_recorders = []

        self._wall_watch = wall_watch

        self._elapsed_time = 0.0

        self._run = False

    def __call__(self) -> None:
        if self._run:
            raise RuntimeError("The evaluator can only be run once.")

        self._run = True

        log.info("Running evaluation on {} device(s).", self._root_gang.size)

        try:
            self._do_run()
            self._publish_evaluation_metrics()
        except KeyboardInterrupt:
            log.info("Evaluation terminated")

            raise

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Evaluation complete in {:,} seconds", int(elapsed_time))

    def _do_run(self) -> None:
        with create_rich_progress() as progress:
            eval_task = progress.add_task("eval", total=None)

            watch = Stopwatch(start=True, device=self._root_gang.device)

            for step_nr in count(start=1):
                self._step_nr = step_nr

                try:
                    batches = next(self._data_reader)
                except StopIteration:
                    break

                progress.update(eval_task, refresh=True, advance=1)

                log.debug("Running step {}.", step_nr)

                for batch in batches:
                    inputs, targets = self._preprocessor(batch)
                    outputs = self._model(inputs)
                    predictions, references = self._postprocessor(outputs, targets)

                    self._metrics.add_batch(
                        predictions=predictions, references=references
                    )

                self._root_gang.barrier()

            self._elapsed_time = watch.get_elapsed_time()

    def _publish_evaluation_metrics(self) -> None:
        """
        publish evaluation metrics to log and TensorBoard folder.
        Note that contrast to fairseq2.metrics, which rely on torcheval,

        HuggingFace's evaluate has an internal support for distributed
        evaluation (see
        https://huggingface.co/docs/evaluate/en/a_quick_tour#distributed-evaluation),
        so we do not to call explicitly sync_and_compute_metrics(), but simply
        evaluate.compute()
        """
        values = self._metrics.compute()

        # In all other rank, values will be zero
        if self._tp_gang.rank != 0 or self._dp_gang.rank != 0:
            return

        assert values is not None

        values["elapsed_time"] = self._elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        record_metrics(self._metric_recorders, "eval", values)

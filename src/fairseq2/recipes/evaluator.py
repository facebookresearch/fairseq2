# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, final

from torch import Tensor

from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.recipes.criterion import Criterion
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


class Evaluator(ABC):
    """Evaluates a machine learning model."""

    @abstractmethod
    def __call__(self) -> None:
        """Run evaluation."""


BatchT = TypeVar("BatchT")


@final
class StandardEvaluator(Evaluator, Generic[BatchT]):
    """Evaluates a machine learning model with a common set of features."""

    _criterion: Criterion[BatchT]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _data_reader: DataReader[BatchT]
    _step_nr: int
    _metric_recorders: List[MetricRecorder]
    _wall_watch: Stopwatch
    _eval_time: float

    def __init__(
        self,
        criterion: Criterion[BatchT],
        gang: Gang,
        data_reader: DataReader[BatchT],
        wall_watch: Stopwatch,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        tb_dir: Optional[Path] = None,
    ) -> None:
        """
        :param criterion:
            The criterion for loss computation.
        :param gang:
            The gang to use for distributed evaluation.
        :param data_reader:
            The data reader of the eval split.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models
            such as LLaMA 70B.
        :param tb_dir:
            The TensorBoard log directory to dump metrics.
        """
        criterion.model.eval()

        self._criterion = criterion

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

        self._step_nr = 0

        if self._tp_gang.rank == 0 and self._dp_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if tb_dir is not None:
                self._metric_recorders.append(TensorBoardRecorder(tb_dir))
        else:
            self._metric_recorders = []

        self._wall_watch = wall_watch

        self._eval_time = 0.0

    @override
    def __call__(self) -> None:
        log.info("Running evaluation on {} device(s).", self._root_gang.size)

        with create_rich_progress() as progress:
            eval_task = progress.add_task("eval", total=None)

            watch = Stopwatch(start=True, device=self._root_gang.device)

            for batches in self._data_reader:
                self._step_nr += 1

                progress.update(eval_task, refresh=True, advance=1)

                log.debug("Running evaluation step {}.", self._step_nr)

                self._criterion.set_step(self._step_nr)

                for batch in batches:
                    self._criterion.compute_loss(batch)

            self._eval_time = watch.get_elapsed_time()

        if self._tp_gang.rank == 0:
            self._publish_evaluation_metrics()

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Evaluation complete in {:,} seconds after {} steps!", int(elapsed_time), self._step_nr)  # fmt: skip

    def _publish_evaluation_metrics(self) -> None:
        metric_bag = self._criterion.valid_metric_bag

        values = metric_bag.sync_and_compute_metrics()

        if self._dp_gang.rank != 0:
            return

        assert values is not None

        self._set_elements_per_second(values, self._eval_time)

        values["elapsed_time"] = self._eval_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        record_metrics(self._metric_recorders, "eval", values, self._step_nr)

    def _set_elements_per_second(
        self, metric_values: Dict[str, Any], elapsed_time: float
    ) -> None:
        try:
            num_elements = metric_values[self._criterion.throughput_metric_name]
        except KeyError:
            return

        if not isinstance(num_elements, (int, float, Tensor)):
            return

        if elapsed_time == 0.0:
            metric_values["elements_per_second"] = 0.0
        else:
            metric_values["elements_per_second"] = num_elements / elapsed_time

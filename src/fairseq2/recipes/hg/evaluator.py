# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any, Generic, TypeVar, final

import torch

from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.models.model import Model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT")


@final
class HFEvaluator(Generic[BatchT]):
    """Evaluate a machine learning model with HuggingFace's evaluate.Metric library"""

    _model: Model
    _preprocessor: Callable[[BatchT], tuple[SequenceBatch, SequenceBatch]]
    _postprocessor: Callable[[Any, SequenceBatch], tuple[list[str], list[str]]]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _data_reader: DataReader[BatchT]
    _metric_recorders: list[MetricRecorder]
    _wall_watch: Stopwatch
    _elapsed_time: float
    _run: bool

    def __init__(
        self,
        model: Model,
        metrics: list[str],
        gang: Gang,
        data_reader: DataReader[BatchT],
        wall_watch: Stopwatch,
        preprocessor: Callable[[BatchT], tuple[SequenceBatch, SequenceBatch]],
        postprocessor: Callable[[Any, SequenceBatch], tuple[list[str], list[str]]],
        dp_gang: Gang | None = None,
        tp_gang: Gang | None = None,
        tb_dir: Path | None = None,
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

                    del inputs
                    del targets
                    del outputs
                    torch.cuda.empty_cache()

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

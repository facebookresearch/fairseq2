# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, final

import torch
from torch.nn import Module

from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    FileMetricRecorder,
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    record_metrics,
)
from fairseq2.recipes.common_metrics import compute_throughput
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import CPU, override
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT")

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class GeneratorUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Generator`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> None:
        """Process ``batch``."""

    @property
    @abstractmethod
    def model(self) -> Module:
        """The underlying model."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The generation-related metrics."""

    @property
    @abstractmethod
    def throughput_metric_name(self) -> Optional[str]:
        """The name of the metric to use for throughput calculation."""


class AbstractGeneratorUnit(GeneratorUnit[BatchT]):
    """Provides a skeletal implementation of :class:`GeneratorUnit`."""

    def __init__(self, model: Module) -> None:
        self._model = model

    @final
    @property
    @override
    def model(self) -> Module:
        return self._model

    @property
    @override
    def throughput_metric_name(self) -> Optional[str]:
        return "num_elements"


@final
class Generator(Generic[BatchT]):
    """Generates output using a machine learning model."""

    _unit: GeneratorUnit[BatchT]
    _data_reader: DataReader[BatchT]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _metric_recorders: List[MetricRecorder]
    _seed: int
    _wall_watch: Stopwatch
    _run: bool

    def __init__(
        self,
        *,
        unit: GeneratorUnit[BatchT],
        data_reader: DataReader[BatchT],
        root_gang: Gang,
        wall_watch: Stopwatch,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        metrics_dir: Optional[Path] = None,
        seed: int = 2,
    ) -> None:
        """
        :param unit:
            The generator unit.
        :param data_reader:
            The data reader.
        :param root_gang:
            The gang for distributed generation.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models
            such as LLaMA 70B.
        :param metrics_dir:
            The directory to dump metrics.
        :param seed:
            The random number generator seed.
        """
        self._unit = unit

        self._data_reader = data_reader

        self._root_gang = root_gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = root_gang
            self._tp_gang = FakeGang(device=root_gang.device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        if self._tp_gang.rank == 0 and self._dp_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if metrics_dir is not None:
                self._metric_recorders.append(FileMetricRecorder(metrics_dir))
        else:
            self._metric_recorders = []

        self._seed = seed

        self._wall_watch = wall_watch

        self._run = False

    @torch.inference_mode()
    def __call__(self) -> None:
        if self._run:
            raise RuntimeError("The generator can only be run once.")

        self._run = True

        log.info("Running generation on {} device(s).", self._root_gang.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Generation terminated!")

            raise

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Generation complete in {:,} seconds!", int(elapsed_time))

    def _do_run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._root_gang.device)

        rng_bag.manual_seed(self._seed)

        watch = Stopwatch(start=True, device=self._root_gang.device)

        self._unit.model.eval()

        with create_rich_progress() as progress:
            task = progress.add_task("generate", total=None)

            for step_nr in count(start=1):
                progress.update(task, advance=1)

                log.debug("Running step {}.", step_nr)

                try:
                    batches = next(self._data_reader)
                except StopIteration:
                    break

                for batch in batches:
                    self._unit(batch)

                self._root_gang.barrier()

        self._publish_metrics(watch.get_elapsed_time())

    def _publish_metrics(self, elapsed_time: float) -> None:
        log.debug("Syncing metrics.")

        if self._tp_gang.rank != 0:
            return

        values = self._unit.metric_bag.sync_and_compute_metrics()

        if self._dp_gang.rank != 0:
            return

        assert values is not None

        compute_throughput(values, self._unit.throughput_metric_name, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        record_metrics(self._metric_recorders, "generate", values)

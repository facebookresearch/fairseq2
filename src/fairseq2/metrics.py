# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from logging import Logger
from pathlib import Path
from string import capwords
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.gang import Gang
from fairseq2.typing import override
from fairseq2.utils.logging import LogWriter, get_log_writer
from fairseq2.utils.profiler import Stopwatch


class MetricBag:
    """Holds a collection of training or validation metrics."""

    _gang: Gang
    _metrics: Dict[str, Metric[Any]]
    _persistent_metrics: Dict[str, Metric[Any]]
    _wall_time: Optional[Stopwatch]

    def __init__(self, gang: Gang, wall_time: Optional[Stopwatch] = None) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        :param wall_time:
            The :class:`Stopwatch` to keep track of process wall time.
        """
        super().__setattr__("_metrics", {})
        super().__setattr__("_persistent_metrics", {})

        self._gang = gang
        self._wall_time = wall_time

    def __getattr__(self, name: str) -> Any:
        if "_metrics" in self.__dict__ and name in self._metrics:
            return self._metrics[name]

        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Metric):
            self.register_metric(name, value, persistent=True)
        else:
            if name in self._metrics:
                del self._metrics[name]

                if name in self._persistent_metrics:
                    del self._persistent_metrics[name]

            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._metrics:
            del self._metrics[name]

            if name in self._persistent_metrics:
                del self._persistent_metrics[name]
        else:
            super().__delattr__(name)

    @final
    def register_metric(
        self, name: str, metric: Metric[Any], persistent: bool = True
    ) -> None:
        """Add ``metric`` to the bag.

        :param name:
            The attribute name to refer to ``metric``.
        :param metric:
            The metric to add.
        :param persistent:
            If ``True``, the state of ``metric`` will be preserved in ``state_dict``.
        """
        if hasattr(self, name):
            raise AttributeError(
                f"`{type(self).__name__}` object already has an attribute '{name}'."
            )

        metric.to(self._gang.device)

        self._metrics[name] = metric

        if persistent:
            self._persistent_metrics[name] = metric

    @final
    def reset_metrics(self) -> None:
        """Reset the metrics to their initial state."""
        for metric in self._metrics.values():
            metric.reset()

    @final
    def sync_and_compute_metrics(self) -> Optional[Dict[str, Any]]:
        """Sync the metrics across all processes and and compute their values."""
        return sync_and_compute_metrics(self)

    def process_metric_values(self, values: Dict[str, Any]) -> None:
        """Process metric ``values``."""
        if self._wall_time is not None:
            values["wall_time"] = self._wall_time.get_elapsed_time()

    @final
    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        for name, metric in self._persistent_metrics.items():
            state_dict[name] = metric.state_dict()

        return state_dict

    @final
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self._persistent_metrics.keys() != state_dict.keys():
            raise ValueError(
                f"`state_dict` must contain metrics {list(self._persistent_metrics.keys())}, but contains {list(state_dict.keys())} instead."
            )

        for name, metric in self._persistent_metrics.items():
            metric.load_state_dict(state_dict[name])

            metric.to(self._gang.device)


def reset_metrics(*bags: MetricBag) -> None:
    """Reset the metrics in ``bags``."""
    for bag in bags:
        bag.reset_metrics()


def sync_and_compute_metrics(*bags: MetricBag) -> Optional[Dict[str, Any]]:
    """Sync the metrics in ``bags`` across all processes and and compute their value."""
    if not bags:
        return None

    gang = bags[0]._gang

    if len(bags) == 1:
        all_metrics = bags[0]._metrics
    else:
        all_metrics = {}

        for bag in bags:
            if bag._gang is not gang:
                raise ValueError("All metric bags in `bags` must use the same gang.")

            all_metrics.update(bag._metrics)

    if gang.size == 1:
        values = {name: m.compute() for name, m in all_metrics.items()}
    else:
        values = sync_and_compute_collection(all_metrics, gang.as_process_group())

    if gang.rank == 0:
        assert values is not None

        for bag in bags:
            bag.process_metric_values(values)

    return values


def format_as_int(value: Any, *, postfix: Optional[str] = None) -> str:
    """Format metric ``value`` as integer."""
    i = int(value)

    s = "<1" if i == 0 and isinstance(value, float) else f"{i:,}"

    if postfix:
        s += postfix

    return s


format_as_seconds = partial(format_as_int, postfix="s")
"""Format metric ``value`` as duration in seconds."""


def format_as_float(value: Any, *, postfix: Optional[str] = None) -> str:
    """Format metric ``value`` as float."""
    s = f"{float(value):g}"

    if postfix:
        s += postfix

    return s


_metric_formatters: Dict[str, Tuple[str, int, Callable[[Any], str]]] = {
    # fmt: off
    "ctc_loss":            ("CTC Loss",                  100, format_as_float),
    "nll_loss":            ("NLL Loss",                  100, format_as_float),
    "uer":                 ("Unit Error Rate (UER)",     200, format_as_float),
    "wer":                 ("Word Error Rate (WER)",     200, format_as_float),
    "elapsed_time":        ("Elapsed Time",              500, format_as_seconds),
    "wall_time":           ("Wall Time",                 510, format_as_seconds),
    "lr":                  ("Learning Rate",             800, format_as_float),
    "loss_scale":          ("Loss Scale",                810, format_as_float),
    "batch_size":          ("Batch Size",                900, format_as_int),
    "elements_per_batch":  ("Elements per Batch",        900, format_as_int),
    "elements_per_second": ("Elements per Second",       900, format_as_int),
    "num_examples":        ("Number of Examples",        900, format_as_int),
    "num_source_elements": ("Number of Source Elements", 900, format_as_int),
    "num_target_elements": ("Number of Target Elements", 900, format_as_int),
    # fmt: on
}


def register_metric_formatter(
    name: str, display_name: str, priority: int, format_fn: Callable[[Any], str]
) -> None:
    """Register a string formatter for the specified metric.

    :param name:
        The name of the metric.
    :param display_name:
        The display name of the metric.
    :param priority:
        The display priority of the metric.
    :param format_fn:
        The callable to convert a metric value to its string representation.
    """
    if name in _metric_formatters:
        raise ValueError(
            f"`name` must be a unique metric name, but '{name}' is already registered."
        )

    _metric_formatters[name] = (display_name, priority, format_fn)


class MetricRecorder(ABC):
    """Records metric values."""

    @abstractmethod
    def record_metrics(
        self,
        run: str,
        values: Dict[str, Any],
        step_nr: int,
        *,
        flush: bool = False,
    ) -> None:
        """Record ``values``.

        :param run:
            The name of the run (e.g. 'train', 'eval').
        :param values:
            The metric values.
        :param step_nr:
            The number of the run step.
        :param flush:
            If ``True``, flushes any buffers after recording.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the recorder."""


def record_metrics(
    recorders: Sequence[MetricRecorder],
    run: str,
    values: Dict[str, Any],
    step_nr: int,
    *,
    flush: bool = False,
) -> None:
    """Record ``values`` to ``recorders``.

    :param recorders:
        The recorders to record to.
    :param run:
        The name of the run (e.g. 'train', 'eval').
    :param values:
        The metric values.
    :param step_nr:
        The number of the run step.
    :param flush:
        If ``True``, flushes any buffers after recording.
    """
    for recorder in recorders:
        recorder.record_metrics(run, values, step_nr, flush=flush)


@final
class LogMetricRecorder(MetricRecorder):
    """Logs metric values to a :class:`Logger`."""

    _log: LogWriter

    def __init__(self, log: Union[LogWriter, Logger]) -> None:
        """
        :param log:
            The log writer or logger to use.
        """
        if isinstance(log, LogWriter):
            self._log = log
        else:
            self._log = LogWriter(log)

    @override
    def record_metrics(
        self,
        run: str,
        values: Dict[str, Any],
        step_nr: int,
        *,
        flush: bool = False,
    ) -> None:
        if not self._log.is_enabled_for(logging.INFO):
            return

        values_and_formatters = []

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                formatter = (name, 999, str)

            values_and_formatters.append((value, formatter))

        # Sort by priority and display name.
        values_and_formatters.sort(key=lambda e: (e[1][1], e[1][0]))

        formatted_values = []

        for value, (display_name, _, fn) in values_and_formatters:
            formatted_values.append(f"{display_name}: {fn(value)}")

        s = " | ".join(formatted_values)

        self._log.info("{} Metrics (step {}) - {}", capwords(run), step_nr, s)

    @override
    def close(self) -> None:
        pass


try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
except ImportError:
    has_tensorboard = False
else:
    has_tensorboard = True


@final
class TensorBoardRecorder(MetricRecorder):
    """Records metric values to TensorBoard."""

    _log_dir: Path
    _writers: Dict[str, SummaryWriter]

    def __init__(self, log_dir: Path) -> None:
        """
        :param log_dir:
            The base directory under which to store the TensorBoard files.
        """
        if not has_tensorboard:
            log = get_log_writer(__name__)

            log.warning("tensorboard not found. Please install it with `pip install tensorboard`.")  # fmt: skip

        self._log_dir = log_dir

        self._writers = {}

    @override
    def record_metrics(
        self,
        run: str,
        values: Dict[str, Any],
        step_nr: int,
        *,
        flush: bool = False,
    ) -> None:
        writer = self._get_writer(run)
        if writer is None:
            return

        for name, value in values.items():
            pair = _metric_formatters.get(name)
            if pair is None:
                display_name = name
            else:
                display_name = pair[0]

            writer.add_scalar(display_name, value, step_nr)

        if flush:
            writer.flush()

    def _get_writer(self, run: str) -> Optional[SummaryWriter]:
        if not has_tensorboard:
            return None

        try:
            writer = self._writers[run]
        except KeyError:
            writer = SummaryWriter(self._log_dir.joinpath(run))

            self._writers[run] = writer

        return writer

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from string import capwords
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union, final

from fairseq2.logging import LogWriter, get_log_writer
from fairseq2.typing import override


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


@dataclass
class _MetricFormatter:
    display_name: str
    priority: int
    fn: Callable[[Any], str]
    log: bool = True


_metric_formatters: Dict[str, _MetricFormatter] = {
    # fmt: off
    "ctc_loss":                      _MetricFormatter("CTC Loss",                        100, format_as_float),
    "nll_loss":                      _MetricFormatter("NLL Loss",                        100, format_as_float),
    "uer":                           _MetricFormatter("Unit Error Rate (UER)",           200, format_as_float),
    "wer":                           _MetricFormatter("Word Error Rate (WER)",           200, format_as_float),
    "gradient_norm":                 _MetricFormatter("Gradient Norm",                   300, format_as_float),
    "elapsed_time":                  _MetricFormatter("Elapsed Time",                    500, format_as_seconds),
    "wall_time":                     _MetricFormatter("Wall Time",                       510, format_as_seconds),
    "lr":                            _MetricFormatter("Learning Rate",                   700, format_as_float),
    "loss_scale":                    _MetricFormatter("Loss Scale",                      710, format_as_float),

    # Batch Metrics
    "batch_size":                    _MetricFormatter("Batch Size",                      800, format_as_int),
    "elements_per_batch":            _MetricFormatter("Elements per Batch",              800, format_as_int),
    "elements_per_second":           _MetricFormatter("Elements per Second",             810, format_as_int),
    "num_examples":                  _MetricFormatter("Number of Examples",              820, format_as_int),
    "num_elements":                  _MetricFormatter("Number of Elements",              830, format_as_int),
    "num_source_elements":           _MetricFormatter("Number of Source Elements",       830, format_as_int),
    "num_target_elements":           _MetricFormatter("Number of Target Elements",       830, format_as_int),
    "total_num_examples":            _MetricFormatter("Total Number of Examples",        840, format_as_int),
    "total_num_elements":            _MetricFormatter("Total Number of Elements",        850, format_as_int),
    "total_num_source_elements":     _MetricFormatter("Total Number of Source Elements", 850, format_as_int),
    "total_num_target_elements":     _MetricFormatter("Total Number of Target Elements", 850, format_as_int),

    # Sequence Generator Metrics
    "generator_prefill_size":        _MetricFormatter("Generator/Prefill Size",          900, format_as_int),
    "generator_num_elements":        _MetricFormatter("Generator/Number of Elements",    901, format_as_int),
    "generator_elements_per_second": _MetricFormatter("Generator/Elements per Second",   902, format_as_int),
    # fmt: on
}


def register_metric_formatter(
    name: str,
    display_name: str,
    priority: int,
    fn: Callable[[Any], str],
    *,
    log: bool = True,
    overwrite: bool = False,
) -> None:
    """Register a string formatter for the specified metric.

    :param name:
        The name of the metric.
    :param display_name:
        The display name of the metric.
    :param priority:
        The display priority of the metric.
    :param fn:
        The callable to convert a metric value to its string representation.
    :param log:
        If ``True``, writes the metric value to log output.
    :param overwrite:
        If ``True``, overwrites any existing metric formatter with the same name.
    """
    if name in _metric_formatters and not overwrite:
        raise ValueError(
            f"`name` must be a unique metric name, but '{name}' is already registered."
        )

    _metric_formatters[name] = _MetricFormatter(display_name, priority, fn, log)


class MetricRecorder(ABC):
    """Records metric values."""

    @abstractmethod
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = False,
    ) -> None:
        """Record ``values``.

        :param run:
            The name of the run (e.g. 'train', 'eval').
        :param values:
            The metric values.
        :param step_nr:
            The step number of the run.
        :param flush:
            If ``True``, flushes any buffers after recording.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the recorder."""


def record_metrics(
    recorders: Sequence[MetricRecorder],
    run: str,
    values: Mapping[str, Any],
    step_nr: Optional[int] = None,
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
        The step number of the run.
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
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = False,
    ) -> None:
        if not self._log.is_enabled_for(logging.INFO):
            return

        values_and_formatters = []

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                formatter = _MetricFormatter(name, 999, str)
            elif not formatter.log:
                continue

            values_and_formatters.append((value, formatter))

        # Sort by priority and display name.
        values_and_formatters.sort(key=lambda e: (e[1].priority, e[1].display_name))

        formatted_values = []

        for value, formatter in values_and_formatters:
            formatted_values.append(f"{formatter.display_name}: {formatter.fn(value)}")

        s = " | ".join(formatted_values)

        if not s:
            s = "N/A"

        if step_nr is None:
            self._log.info("{} Metrics - {}", capwords(run), s)
        else:
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
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = False,
    ) -> None:
        writer = self._get_writer(run)
        if writer is None:
            return

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                display_name = name
            else:
                display_name = formatter.display_name

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from logging import Logger
from pathlib import Path
from string import capwords
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Union,
    final,
)

from torch import Tensor

from fairseq2.logging import LogWriter, get_log_writer
from fairseq2.typing import override


def format_as_int(value: Any, *, postfix: Optional[str] = None) -> str:
    """Format metric ``value`` as integer."""
    try:
        i = int(value)
    except ValueError:
        return f"{value}"

    s = "<1" if i == 0 and isinstance(value, float) else f"{i:,}"

    if postfix:
        s += postfix

    return s


format_as_seconds = partial(format_as_int, postfix="s")
"""Format metric ``value`` as duration in seconds."""


def format_as_float(value: Any, *, postfix: Optional[str] = None) -> str:
    """Format metric ``value`` as float."""
    try:
        s = f"{float(value):g}"
    except ValueError:
        return f"{value}"

    if postfix:
        s += postfix

    return s


_UNITS: Final = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]


def format_as_byte_size(value: Any) -> str:
    """Format metric ``value`` in byte units."""
    unit_idx = 0

    try:
        size = float(value)
    except ValueError:
        return f"{value}"

    if not math.isfinite(size) or size <= 0.0:
        return "0 B"

    while size >= 1024:
        size /= 1024

        unit_idx += 1

    try:
        return f"{size:.2f} {_UNITS[unit_idx]}"
    except IndexError:
        return "TOO BIG"


@dataclass
class _MetricFormatter:
    display_name: str
    priority: int
    fn: Callable[[Any], str] = str
    log: bool = True


_metric_formatters: Dict[str, _MetricFormatter] = {
    # fmt: off
    "ctc_loss":                      _MetricFormatter("CTC Loss",                        100, format_as_float),
    "nll_loss":                      _MetricFormatter("NLL Loss",                        100, format_as_float),
    "bleu":                          _MetricFormatter("BLEU",                            200, format_as_float),
    "chrf":                          _MetricFormatter("chrF++",                          200, format_as_float),
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
    "generator_cache_size":          _MetricFormatter("Generator/Cache Size",            903, format_as_byte_size),
    "generator_cache_capacity":      _MetricFormatter("Generator/Cache Capacity",        904, format_as_byte_size),
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


def format_metric_value(name: str, value: Any) -> str:
    """Format the specified metric along with its value as a string."""
    formatter = _metric_formatters.get(name)
    if formatter is None:
        return f"{name}: {value}"

    return f"{formatter.display_name}: {formatter.fn(value)}"


class MetricRecorder(ABC):
    """Records metric values."""

    @abstractmethod
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = True,
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
    flush: bool = True,
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
        flush: bool = True,
    ) -> None:
        if not self._log.is_enabled_for_info():
            return

        values_and_formatters = []

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                formatter = _MetricFormatter(name, 999)
            elif not formatter.log:
                continue

            values_and_formatters.append((value, formatter))

        # Sort by priority and display name.
        values_and_formatters.sort(key=lambda p: (p[1].priority, p[1].display_name))

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


@final
class JsonFileMetricRecorder(MetricRecorder):
    """Records metric values to JSONL files."""

    _RUN_PART_REGEX: Final = re.compile("^[-_a-zA-Z0-9]+$")

    _output_dir: Path
    _streams: Dict[str, TextIO]

    def __init__(self, output_dir: Path) -> None:
        """
        :param output_dir:
            The base directory under which to store the metric files.
        """
        self._output_dir = output_dir.expanduser().resolve()

        self._streams = {}

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = True,
    ) -> None:
        run = run.strip()

        for part in run.split("/"):
            if re.match(self._RUN_PART_REGEX, part) is None:
                raise ValueError(
                    f"`run` must contain only alphanumeric characters, dash, underscore, and forward slash, but is '{run}' instead."
                )

        stream = self._get_stream(run)

        values_and_formatters = []

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                formatter = _MetricFormatter(name, 999)

            values_and_formatters.append((value, formatter))

        # Sort by priority and display name.
        values_and_formatters.sort(key=lambda p: (p[1].priority, p[1].display_name))

        def sanitize(value: Any, formatter: _MetricFormatter) -> Any:
            if isinstance(value, Tensor):
                value = value.item()

            if formatter.fn is format_as_int:
                try:
                    value = int(value)
                except ValueError:
                    pass

            return value

        output: Dict[str, Any] = {"Time": datetime.utcnow().isoformat()}

        if step_nr is not None:
            output["Step"] = step_nr

        for value, formatter in values_and_formatters:
            output[formatter.display_name] = sanitize(value, formatter)

        json.dump(output, stream, indent=None)

        stream.write("\n")

        if flush:
            stream.flush()

    def _get_stream(self, run: str) -> TextIO:
        try:
            return self._streams[run]
        except KeyError:
            pass

        file = self._output_dir.joinpath(run).with_suffix(".jsonl")

        try:
            file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The metric directory ({file.parent}) cannot be created. See nested exception for details."
            ) from ex

        try:
            fp = file.open("a")
        except OSError as ex:
            raise RuntimeError(
                f"The metric file ({file}) cannot be created. See nested exception for details."
            ) from ex

        self._streams[run] = fp

        return fp

    @override
    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

        self._streams.clear()


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
        flush: bool = True,
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

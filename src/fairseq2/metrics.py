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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.gang import Gang
from fairseq2.typing import finaloverride


class MetricBag:
    """Holds a collection of training or validation metrics."""

    gang: Gang
    metrics: Dict[str, Metric[Any]]
    persistent_metrics: Dict[str, Metric[Any]]

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        """
        super().__setattr__("metrics", {})
        super().__setattr__("persistent_metrics", {})

        self.gang = gang

    def __getattr__(self, name: str) -> Any:
        if "metrics" in self.__dict__ and name in self.metrics:
            return self.metrics[name]

        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Metric):
            self.register_metric(name, value, persistent=True)
        else:
            if name in self.metrics:
                del self.metrics[name]

                if name in self.persistent_metrics:
                    del self.persistent_metrics[name]

            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.metrics:
            del self.metrics[name]

            if name in self.persistent_metrics:
                del self.persistent_metrics[name]
        else:
            super().__delattr__(name)

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

        metric.to(self.gang.device)

        self.metrics[name] = metric

        if persistent:
            self.persistent_metrics[name] = metric

    def reset_metrics(self) -> None:
        """Reset the metrics to their initial state."""
        for metric in self.metrics.values():
            metric.reset()

    def sync_and_compute_metrics(self) -> Optional[Dict[str, Any]]:
        """Sync the metrics across all processes and and compute their values."""
        return sync_and_compute_metrics(self)

    def process_metric_values(self, values: Dict[str, Any]) -> None:
        """Process metric ``values``."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        for name, metric in self.persistent_metrics.items():
            state_dict[name] = metric.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self.persistent_metrics.keys() != state_dict.keys():
            raise ValueError(
                f"`state_dict` must contain metrics {list(self.persistent_metrics.keys())}, but contains {list(state_dict.keys())} instead."
            )

        for name, metric in self.persistent_metrics.items():
            metric.load_state_dict(state_dict[name])

            metric.to(self.gang.device)


def reset_metrics(*bags: MetricBag) -> None:
    """Reset the metrics in ``bags``."""
    for bag in bags:
        bag.reset_metrics()


def sync_and_compute_metrics(*bags: MetricBag) -> Optional[Dict[str, Any]]:
    """Sync the metrics in ``bags`` across all processes and and compute their value."""
    if not bags:
        return None

    gang = bags[0].gang

    if len(bags) == 1:
        all_metrics = bags[0].metrics
    else:
        all_metrics = {}

        for bag in bags:
            if bag.gang is not gang:
                raise ValueError("All metric bags in `bags` must use the same gang.")

            all_metrics.update(bag.metrics)

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


def format_as_float(
    value: Any, *, decimal: Optional[int] = None, postfix: Optional[str] = None
) -> str:
    """Format metric ``value`` as float."""
    if decimal:
        s = f"{float(value):,.{decimal}f}"
    else:
        s = f"{float(value):,}"

    if postfix:
        s += postfix

    return s


format_as_loss = partial(format_as_float, decimal=3)
"""Format metric ``value`` as training loss."""


_metric_formatters: Dict[str, Tuple[str, Callable[[Any], str]]] = {
    # fmt: off
    "batch_size":          ("Batch Size",                format_as_int),
    "elapsed_time":        ("Elapsed Time",              format_as_seconds),
    "elements_per_batch":  ("Elements per Batch",        format_as_int),
    "elements_per_second": ("Elements per Second",       format_as_int),
    "entropy_loss":        ("Entropy Loss",              format_as_loss),
    "grad_scale":          ("Grad Scale",                format_as_float),
    "loss":                ("Loss",                      format_as_loss),
    "lr":                  ("Learning Rate",             format_as_float),
    "num_source_elements": ("Number of Source Elements", format_as_int),
    "num_target_elements": ("Number of Target Elements", format_as_int),
    "wall_time":           ("Wall Time",                 format_as_seconds),
    # fmt: on
}


def register_metric_formatter(
    name: str, display_name: str, formatter: Callable[[Any], str]
) -> None:
    """Register a string formatter for the specified metric.

    :param name:
        The name of the metric.
    :param display_name:
        The display name of the metric.
    :param formatter:
        The formatter to convert a metric value to its string representation.
    """
    if name in _metric_formatters:
        raise ValueError(
            f"`name` must be a unique metric name, but '{name}' is already registered."
        )

    _metric_formatters[name] = (display_name, formatter)


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

    def close(self) -> None:
        """Close the recorder."""
        pass


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

    logger: Logger

    def __init__(self, logger: Logger) -> None:
        """
        :param logger:
            The logger to use.
        """
        self.logger = logger

    @finaloverride
    def record_metrics(
        self,
        run: str,
        values: Dict[str, Any],
        step_nr: int,
        *,
        flush: bool = False,
    ) -> None:
        if not self.logger.isEnabledFor(logging.INFO):
            return

        formatted_values = []

        for name, value in values.items():
            pair = _metric_formatters.get(name)
            if pair is None:
                formatted_values.append(f"{name}: {value}")
            else:
                display_name, formatter = pair

                formatted_values.append(f"{display_name}: {formatter(value)}")

        s = " | ".join(formatted_values)

        self.logger.info(f"{run} Metrics (step {step_nr}) - {s}")


try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
except ImportError:
    has_tensorboard = False
else:
    has_tensorboard = True


@final
class TensorBoardRecorder(MetricRecorder):
    """Records metric values to TensorBoard."""

    log_dir: Path

    _writers: Dict[str, SummaryWriter]

    def __init__(self, log_dir: Path) -> None:
        """
        :param log_dir:
            The base directory under which to store the TensorBoard files.
        """
        if not has_tensorboard:
            logger = logging.getLogger(__name__)

            logger.warning("tensorboard not found. Please install it with `pip install tensorboard`.")  # fmt: skip

        self.log_dir = log_dir

        self._writers = {}

    @finaloverride
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
            writer = SummaryWriter(self.log_dir.joinpath(run))

            self._writers[run] = writer

        return writer

    @finaloverride
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()

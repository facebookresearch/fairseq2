# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.gang import Gang


class MetricBag:
    """Holds a collection of training or validation metrics."""

    gang: Gang
    training: bool
    metrics: Dict[str, Metric[Any]]
    persistent_metrics: Dict[str, Metric[Any]]

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang to use to sync metrics across ranks.
        """
        super().__setattr__("metrics", {})
        super().__setattr__("persistent_metrics", {})

        self.gang = gang
        self.training = True

    def __getattr__(self, name: str) -> Any:
        if "metrics" in self.__dict__ and name in self.metrics:
            return self.metrics[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Metric):
            if "metrics" not in self.__dict__:
                raise AttributeError("`__init__()` must be called first.")

            self.register_metric(name, value, persistent=True)
        else:
            if name in self.metrics:
                del self.metrics[name]

                if name in self.persistent_metrics:
                    del self.persistent_metrics[name]

            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if "metrics" in self.__dict__ and name in self.metrics:
            del self.metrics[name]

            if name in self.persistent_metrics:
                del self.persistent_metrics[name]
        else:
            super().__delattr__(name)

    def register_metric(
        self, name: str, metric: Metric[Any], persistent: bool = True
    ) -> None:
        """Register a metric with the bag.

        :param name:
            The name of the metric.
        :param metric:
            The :class:`torcheval.metrics.Metric` instance.
        :param persistent:
            If ``True``, the metric state will be preserved in ``state_dict``.
        """
        metric.to(self.gang.device)

        self.metrics[name] = metric

        if persistent:
            self.persistent_metrics[name] = metric

    def reset_metrics(self) -> None:
        """Reset the metrics to their initial state."""
        for metric in self.metrics.values():
            metric.reset()

    def sync_and_compute_metrics(self) -> Optional[Dict[str, Any]]:
        """Sync the metrics across all ranks and and compute their value."""
        return sync_and_compute_metrics(self)

    def train(self, mode: bool = True) -> None:
        """Set the bag to training or validation mode.

        :param model:
            If ``True``, sets to training mode; otherwise, to validation mode.
        """
        self.training = True

    def valid(self) -> None:
        """Set the bag to validation mode."""
        self.train(False)

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

    def format_metric_values(self, values: Dict[str, Any]) -> str:
        """Format metric ``values`` to print to stdout or stderr."""
        return format_metric_values(values, self)

    def format_metric_value(self, name: str, value: Any) -> Optional[str]:
        """Format the value of metric ``name`` to print to stdout or stderr."""
        return None


def reset_metrics(*bags: MetricBag) -> None:
    """Reset the metrics in ``bags``."""
    for bag in bags:
        bag.reset_metrics()


def sync_and_compute_metrics(*bags: MetricBag) -> Optional[Dict[str, Any]]:
    """Sync the metrics in ``bags`` across all ranks and and compute their value."""
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

    return sync_and_compute_collection(all_metrics, gang.as_process_group())


def format_metric_values(values: Dict[str, Any], *bags: MetricBag) -> str:
    """Format metric ``values`` to print to stdout or stderr."""
    fmt_values = []

    for name, value in values.items():
        for bag in bags:
            fmt_value = bag.format_metric_value(name, value)
            if fmt_value is not None:
                break

        if fmt_value is None:
            fmt_value = str(value)

        fmt_values.append(f"{name}: {fmt_value}")

    return " | ".join(fmt_values)

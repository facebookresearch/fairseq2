# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.gang import Gang
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
            The gang over which to sync metrics.
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
        """Sync the metrics across all processes and compute their values."""
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
    """Sync the metrics across all processes and and compute their values."""
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

    logging.disable(logging.WARNING)  # Suppress "No calls to update()".

    try:
        if gang.size == 1:
            values = {name: m.compute() for name, m in all_metrics.items()}
        else:
            values = sync_and_compute_collection(all_metrics, gang.as_process_group())
    finally:
        logging.disable(logging.NOTSET)

    if gang.rank == 0:
        assert values is not None

        for bag in bags:
            bag.process_metric_values(values)

    return values

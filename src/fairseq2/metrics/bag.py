# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Sequence, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.gang import Gang


class MetricBag:
    """Holds a collection of training or validation metrics."""

    _gang: Gang
    _metrics: Dict[str, Metric[Any]]
    _persistent_metrics: Dict[str, Metric[Any]]
    _original_metrics: Optional[Dict[str, Metric[Any]]]

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__setattr__("_metrics", {})
        super().__setattr__("_persistent_metrics", {})
        super().__setattr__("_original_metrics", None)

        self._gang = gang

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
    def begin_updates(self) -> None:
        """Begin a transactional update of multiple metrics.

        A call to ``begin_updates()`` must be followed by a ``commit_updates()``
        or ``rollback_updates()``.
        """
        if self._original_metrics is not None:
            raise ValueError("`begin_updates()` has already been called.")

        self._original_metrics = deepcopy(self._metrics)

    @final
    def commit_updates(self) -> None:
        """Commit pending metric updates."""
        if self._original_metrics is None:
            raise ValueError("`begin_updates()` must be called first.")

        self._original_metrics = None

    @final
    def rollback_updates(self) -> None:
        """Discard pending metric updates and rollback to the original state."""
        if self._original_metrics is None:
            raise ValueError("`begin_updates()` must be called first.")

        self._metrics, self._original_metrics = self._original_metrics, None

        for name, metric in self._metrics.items():
            if name in self._persistent_metrics:
                self._persistent_metrics[name] = metric

    @final
    def reset_metrics(self) -> None:
        """Reset the metrics to their initial state."""
        for metric in self._metrics.values():
            metric.reset()

    @final
    def reset_non_persistent_metrics(self) -> None:
        """Reset the non-persistent metrics to their initial state."""
        for name, metric in self._metrics.items():
            if name not in self._persistent_metrics:
                metric.reset()

    @final
    def sync_and_compute_metrics(self) -> Optional[Dict[str, Any]]:
        """Sync the metrics across all processes and compute their values."""
        return sync_and_compute_metrics([self])

    def process_metric_values(self, values: Dict[str, Any]) -> None:
        """Process metric ``values``."""

    @property
    def metrics(self) -> Mapping[str, Metric[Any]]:
        """The metrics contained in this bag."""
        return self._metrics

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


def reset_metrics(bags: Sequence[MetricBag]) -> None:
    """Reset the metrics in ``bags``."""
    for bag in bags:
        bag.reset_metrics()


def reset_non_persistent_metrics(bags: Sequence[MetricBag]) -> None:
    """Reset the non-persistent metrics in ``bags``."""
    for bag in bags:
        bag.reset_non_persistent_metrics()


def sync_and_compute_metrics(bags: Sequence[MetricBag]) -> Optional[Dict[str, Any]]:
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

    try:
        logging.disable(logging.WARNING)  # Suppress "No calls to update()".

        if gang.size == 1:
            values = {name: m.compute() for name, m in all_metrics.items()}
        else:
            values = sync_and_compute_collection(all_metrics, gang.as_process_group())
    finally:
        logging.disable(logging.NOTSET)

    if gang.rank == 0:
        assert values is not None

        def strip_underscore(s: str) -> str:
            if s.startswith("_"):
                s = s[1:]

            return s

        values = {strip_underscore(n): v for n, v in values.items()}

        for bag in bags:
            bag.process_metric_values(values)

    return values


def merge_metric_states(
    sources: Mapping[str, Metric[Any]], targets: Mapping[str, Metric[Any]]
) -> None:
    """Merge the states of the same-named metrics from ``sources`` to ``targets``."""
    for name, target_metric in targets.items():
        try:
            source_metric = sources[name]
        except KeyError:
            continue

        if type(target_metric) is type(source_metric):
            target_metric.merge_state([source_metric])

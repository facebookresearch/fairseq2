# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection

from fairseq2.device import Device
from fairseq2.error import ContractError, InternalError, InvalidOperationError
from fairseq2.gang import Gang


class MetricBag:
    """Holds a collection of training or validation metrics."""

    _metrics: dict[str, Metric[Any]]
    _original_metrics: dict[str, Metric[Any]] | None

    def __init__(self) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__setattr__("_metrics", {})
        super().__setattr__("_original_metrics", None)

    def __getattr__(self, name: str) -> Any:
        if "_metrics" in self.__dict__ and name in self._metrics:
            return self._metrics[name]

        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Metric):
            self.register_metric(name, value)
        else:
            if name in self._metrics:
                del self._metrics[name]

            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._metrics:
            del self._metrics[name]
        else:
            super().__delattr__(name)

    @final
    def register_metric(self, name: str, metric: Metric[Any]) -> None:
        """
        Adds ``metric`` to the bag.

        :param name: The attribute name to refer to ``metric``.
        :param metric: The metric to add.
        """
        if hasattr(self, name):
            raise AttributeError(
                f"`{type(self).__name__}` object already has an attribute '{name}'."
            )

        self._metrics[name] = metric

    @final
    def begin_updates(self) -> None:
        """
        Begins a transactional update of multiple metrics.

        A call to ``begin_updates()`` must be followed by a ``commit_updates()``
        or ``rollback_updates()``.
        """
        if self._original_metrics is not None:
            raise InvalidOperationError("`begin_updates()` has already been called.")

        try:
            self._original_metrics = deepcopy(self._metrics)
        except Exception as ex:
            raise ContractError(
                "The metrics in the bag cannot be copied. See the nested exception for details."
            ) from ex

    @final
    def commit_updates(self) -> None:
        """Commits pending metric updates."""
        if self._original_metrics is None:
            raise InvalidOperationError("`begin_updates()` must be called first.")

        self._original_metrics = None

    @final
    def rollback_updates(self) -> None:
        """Discards pending metric updates and rollback to the original state."""
        if self._original_metrics is None:
            raise InvalidOperationError("`begin_updates()` must be called first.")

        self._metrics, self._original_metrics = self._original_metrics, None

    @final
    def reset_metrics(self) -> None:
        """Resets the metrics to their initial state."""
        for metric in self._metrics.values():
            metric.reset()

    def process_metric_values(self, values: dict[str, object]) -> None:
        """Process metric ``values``."""

    @property
    def metrics(self) -> Mapping[str, Metric[Any]]:
        """The metrics contained in this bag."""
        return self._metrics

    @final
    def state_dict(self) -> dict[str, object]:
        state_dict: dict[str, object] = {}

        for name, metric in self._metrics.items():
            state_dict[name] = metric.state_dict()

        return state_dict

    @final
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        state_keys = set(state_dict.keys())

        metric_names = set(self._metrics.keys())

        if metric_names != state_keys:
            missing_metrics = metric_names - state_keys
            if missing_metrics:
                s = ", ".join(sorted(missing_metrics))

                raise ValueError(
                    f"`state_dict` must contain the states of the following metric(s): {s}"
                )

            extra_keys = state_keys - metric_names
            if extra_keys:
                s = ", ".join(sorted(extra_keys))

                raise ValueError(
                    f"`state_dict` must contain only the states of the metrics of this bag, but it contains the following unexpected key(s): {s}"
                )

        for name, metric in self._metrics.items():
            metric_state_dict = state_dict[name]
            if not isinstance(metric_state_dict, dict):
                raise TypeError(
                    f"`state_dict['{name}']` must be of type `dict`, but is of type `{type(metric_state_dict)}` instead."
                )

            device = metric.device

            try:
                metric.load_state_dict(metric_state_dict)
            except (RuntimeError, ValueError, TypeError) as ex:
                raise ValueError(
                    f"`state_dict['{name}']` is not a valid `{type(metric)}` state. See the nested exception for details."
                ) from ex

            metric.to(device)

    def to(self, device: Device) -> None:
        for metric in self._metrics.values():
            metric.to(device)


def sync_and_compute_metrics(bag: MetricBag, gang: Gang) -> dict[str, object] | None:
    """Sync the metrics across all processes and and compute their values."""
    try:
        # TODO: disable torcheval only??
        logging.disable(logging.WARNING)  # Suppress "No calls to update()".

        if gang.size == 1:
            metric_values = {name: m.compute() for name, m in bag.metrics.items()}
        else:
            metrics = dict(bag.metrics)

            metric_values = sync_and_compute_collection(
                metrics, gang.as_process_group()
            )
    except RuntimeError as ex:
        raise MetricBagError(
            "The metric values cannot be synced. See the nested exception for details."
        ) from ex
    finally:
        logging.disable(logging.NOTSET)

    if gang.rank == 0:
        if metric_values is None:
            raise InternalError("`metric_values` is `None`.")

        bag.process_metric_values(metric_values)

    return metric_values


class MetricBagError(Exception):
    pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypeVar, final

from torcheval.metrics import Metric
from torcheval.metrics.toolkit import sync_and_compute_collection
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import InternalError, InvalidOperationError, StateDictError
from fairseq2.gang import Gang, GangError
from fairseq2.typing import Stateful

MetricT = TypeVar("MetricT", bound=Metric[Any])


@final
class MetricBag(Stateful):
    """Holds a collection of training or validation metrics."""

    def __init__(self, device: Device) -> None:
        self._device = device
        self._metrics: dict[str, Metric[Any]] = {}
        self._original_metrics: dict[str, Metric[Any]] | None = None

    def add(self, name: str, metric: Metric[Any]) -> None:
        if name in self._metrics:
            raise ValueError(f"A metric named {name} is already registered.")

        self._metrics[name] = metric.to(self._device)

    def get(self, name: str, kls: type[MetricT]) -> MetricT:
        metric = self._metrics.get(name)
        if metric is None:
            raise ValueError(f"A metric named {name} is not found.")

        if not isinstance(metric, kls):
            raise TypeError(
                f"{name} metric is expected to be of type `{kls}`, but is of type `{type(metric)}` instead."
            )

        return metric

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
            raise InternalError("Metrics in the bag cannot be copied.") from ex

    def commit_updates(self) -> None:
        """Commits pending metric updates."""
        if self._original_metrics is None:
            raise InvalidOperationError("`begin_updates()` must be called first.")

        self._original_metrics = None

    def rollback_updates(self) -> None:
        """Discards pending metric updates and rollback to the original state."""
        if self._original_metrics is None:
            raise InvalidOperationError("`begin_updates()` must be called first.")

        self._metrics, self._original_metrics = self._original_metrics, None

    def reset_metrics(self) -> None:
        """Resets the metrics to their initial state."""
        for metric in self._metrics.values():
            metric.reset()

    @property
    def metrics(self) -> Mapping[str, Metric[Any]]:
        """The metrics contained in this bag."""
        return self._metrics

    @override
    def state_dict(self) -> dict[str, object]:
        return {n: m.state_dict() for n, m in self._metrics.items()}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        state_dict = dict(state_dict)

        for name, metric in self._metrics.items():
            try:
                metric_state_dict = state_dict.pop(name)
            except KeyError:
                raise StateDictError(
                    f"`state_dict` is expected to contain a key named '{name}'."
                )

            if not isinstance(metric_state_dict, dict):
                raise StateDictError(
                    f"`state_dict['{name}']` is expected to be of type `{dict}`, but is of type `{type(metric_state_dict)}` instead."
                )

            try:
                metric.load_state_dict(metric_state_dict)
            except (RuntimeError, ValueError, TypeError, StateDictError) as ex:
                raise StateDictError(
                    f"`state_dict['{name}']` does not represent a valid `{type(metric)}` state."
                ) from ex

            metric.to(self._device)

        StateDictError.raise_if_not_empty(state_dict)

    @property
    def device(self) -> Device:
        return self._device


def sync_and_compute_metrics(bag: MetricBag, gang: Gang) -> dict[str, object] | None:
    """Sync the metrics across all processes and and compute their values."""
    if gang.device != bag.device:
        raise ValueError("`bag.device` and `gang.device` must be same.")

    if gang.size == 1:
        metric_values = {name: m.compute() for name, m in bag.metrics.items()}
    else:
        metrics = dict(bag.metrics)

        try:
            metric_values = sync_and_compute_collection(
                metrics, gang.as_process_group()
            )
        except RuntimeError as ex:
            raise GangError("Metric value synchronization failed.") from ex

    return metric_values

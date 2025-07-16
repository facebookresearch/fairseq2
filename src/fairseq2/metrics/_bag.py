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

from fairseq2.device import Device
from fairseq2.error import ContractError, InvalidOperationError
from fairseq2.gang import Gang

MetricT = TypeVar("MetricT", bound=Metric[Any])


@final
class MetricBag:
    """Holds a collection of training or validation metrics."""

    _device: Device
    _metrics: dict[str, Metric[Any]]
    _original_metrics: dict[str, Metric[Any]] | None

    def __init__(self, device: Device) -> None:
        self._device = device

        self._metrics = {}

        self._original_metrics = None

    def get(self, kls: type[MetricT], name: str, *args: Any, **kwargs: Any) -> MetricT:
        metric = self._metrics.get(name)
        if metric is not None:
            if not isinstance(metric, kls):
                raise TypeError(
                    f"The '{name}' metric must be of type `{kls}`, but is of type `{type(metric)}` instead."
                )

            return metric

        metric = kls(*args, **kwargs, device=self._device)

        self._metrics[name] = metric

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
            raise ContractError(
                "The metrics in the bag cannot be copied. See the nested exception for details."
            ) from ex

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

    def state_dict(self) -> dict[str, object]:
        state_dict: dict[str, object] = {}

        for name, metric in self._metrics.items():
            state_dict[name] = metric

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._metrics.clear()

        for name, metric in state_dict.items():
            if not isinstance(metric, Metric):
                raise ValueError(
                    f"`state_dict['{name}']` must be of type `{Metric}`, but is of type `{type(metric)}` instead."
                )

            metric.to(self._device)

            self._metrics[name] = metric

    @property
    def device(self) -> Device:
        return self._device


def sync_and_compute_metrics(bag: MetricBag, gang: Gang) -> dict[str, object] | None:
    """Sync the metrics across all processes and and compute their values."""
    if gang.device != bag.device:
        raise ValueError("`bag.device` and `gang.device` must be same.")

    try:
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

    return metric_values


class MetricBagError(Exception):
    pass

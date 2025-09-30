# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, final

from typing_extensions import override

from fairseq2.logging import log
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.recorder import MetricRecorder


@final
class LogMetricRecorder(MetricRecorder):
    """Logs metric values to a :class:`Logger`."""

    _DISPLAY_NAMES: Final = {"valid": "Validation", "eval": "Evaluation"}

    def __init__(self, metric_descriptors: MetricDescriptorRegistry) -> None:
        self._metric_descriptors = metric_descriptors

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        if not log.is_enabled_for_info():
            return

        values_and_descriptors = []

        for name, value in values.items():
            descriptor = self._metric_descriptors.maybe_get(name)
            if descriptor is None:
                descriptor = MetricDescriptor(
                    name, name, 999, formatter=lambda v: str(v)
                )
            elif not descriptor.log:
                continue

            values_and_descriptors.append((value, descriptor))

        # Sort by priority and display name.
        values_and_descriptors.sort(key=lambda p: (p[1].priority, p[1].display_name))

        formatted_values = []

        for value, descriptor in values_and_descriptors:
            formatted_values.append(
                f"{descriptor.display_name}: {descriptor.formatter(value)}"
            )

        s = " | ".join(formatted_values)

        if not s:
            s = "N/A"

        category_parts = category.split("/")

        title = self._DISPLAY_NAMES.get(category_parts[0])
        if title is None:
            title = category_parts[0].capitalize()

        if step_nr is None:
            m = f"{title} Metrics"
        else:
            m = f"{title} Metrics (step {step_nr})"

        if len(category_parts) > 1:
            m = f"{m} - {'/'.join(category_parts[1:])}"

        log.info("{} - {}", m, s)

    @override
    def close(self) -> None:
        pass

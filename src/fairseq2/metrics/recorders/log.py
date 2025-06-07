# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from typing_extensions import override

from fairseq2.logging import LogWriter
from fairseq2.metrics.recorders.descriptor import MetricDescriptor
from fairseq2.metrics.recorders.recorder import MetricRecorder
from fairseq2.typing import Provider


@final
class LogMetricRecorder(MetricRecorder):
    """Logs metric values to a :class:`Logger`."""

    _log: LogWriter
    _metric_descriptors: Provider[MetricDescriptor]
    _display_names: dict[str, str]

    def __init__(
        self, log: LogWriter, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        self._log = log
        self._metric_descriptors = metric_descriptors

        self._display_names = {"valid": "Validation", "eval": "Evaluation"}

    @override
    def record_metric_values(
        self, section: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        if not self._log.is_enabled_for_info():
            return

        values_and_descriptors = []

        for name, value in values.items():
            try:
                descriptor = self._metric_descriptors.resolve(name)
            except LookupError:
                descriptor = None

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

        section_parts = section.split("/")

        title = self._display_names.get(section_parts[0])
        if title is None:
            title = section_parts[0].capitalize()

        if step_nr is None:
            m = f"{title} Metrics"
        else:
            m = f"{title} Metrics (step {step_nr})"

        if len(section_parts) > 1:
            m = f"{m} - {'/'.join(section_parts[1:])}"

        self._log.info("{} - {}", m, s)

    @override
    def close(self) -> None:
        pass

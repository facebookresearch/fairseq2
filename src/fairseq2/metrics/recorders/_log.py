# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.logging import LogWriter
from fairseq2.metrics import MetricDescriptor
from fairseq2.metrics.recorders._handler import MetricRecorderHandler
from fairseq2.metrics.recorders._recorder import MetricRecorder, NoopMetricRecorder
from fairseq2.registry import Provider
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


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
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        if not self._log.is_enabled_for_info():
            return

        values_and_descriptors = []

        for name, value in values.items():
            try:
                descriptor = self._metric_descriptors.get(name)
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

        run_parts = run.split("/")

        phase = self._display_names.get(run_parts[0])
        if phase is None:
            phase = run_parts[0].capitalize()

        if step_nr is None:
            m = f"{phase} Metrics"
        else:
            m = f"{phase} Metrics (step {step_nr})"

        if len(run_parts) > 1:
            m = f"{m} - {'/'.join(run_parts[1:])}"

        self._log.info("{} - {}", m, s)

    @override
    def close(self) -> None:
        pass


LOG_METRIC_RECORDER: Final = "log"


@dataclass(kw_only=True)
class LogMetricRecorderConfig:
    enabled: bool = True


@final
class LogMetricRecorderHandler(MetricRecorderHandler):
    _log: LogWriter
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(
        self, log: LogWriter, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        self._log = log
        self._metric_descriptors = metric_descriptors

    @override
    def create(self, output_dir: Path, config: object) -> MetricRecorder:
        config = structure(config, LogMetricRecorderConfig)

        validate(config)

        if not config.enabled:
            return NoopMetricRecorder()

        return LogMetricRecorder(self._log, self._metric_descriptors)

    @property
    @override
    def name(self) -> str:
        return LOG_METRIC_RECORDER

    @property
    @override
    def config_kls(self) -> type[object]:
        return LogMetricRecorderConfig

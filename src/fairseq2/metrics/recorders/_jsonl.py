# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO, final

from torch import Tensor
from typing_extensions import override

from fairseq2.metrics import MetricDescriptor, format_as_int
from fairseq2.metrics.recorders._handler import MetricRecorderHandler
from fairseq2.metrics.recorders._recorder import (
    MetricRecorder,
    MetricRecordError,
    NoopMetricRecorder,
)
from fairseq2.registry import Provider
from fairseq2.utils.file import FileMode, FileSystem
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class JsonlMetricRecorder(MetricRecorder):
    """Records metric values to JSONL files."""

    _RUN_PART_REGEX: Final = re.compile("^[-_a-zA-Z0-9]+$")

    _output_dir: Path
    _file_system: FileSystem
    _metric_descriptors: Provider[MetricDescriptor]
    _streams: dict[str, TextIO]

    def __init__(
        self,
        output_dir: Path,
        file_system: FileSystem,
        metric_descriptors: Provider[MetricDescriptor],
    ) -> None:
        """
        :param output_dir: The base directory under which to store the metric
            files.
        """
        self._output_dir = output_dir
        self._file_system = file_system
        self._metric_descriptors = metric_descriptors

        self._streams = {}

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        run = run.strip()

        for part in run.split("/"):
            if re.match(self._RUN_PART_REGEX, part) is None:
                raise ValueError(
                    f"`run` must contain only alphanumeric characters, dash, underscore, and forward slash, but is '{run}' instead."
                )

        stream = self._get_stream(run)

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

            values_and_descriptors.append((value, descriptor))

        # Sort by priority and display name.
        values_and_descriptors.sort(key=lambda p: (p[1].priority, p[1].display_name))

        def sanitize(value: object, descriptor: MetricDescriptor) -> object:
            if isinstance(value, Tensor):
                value = value.item()

            if descriptor.formatter is format_as_int:
                if isinstance(value, (str, Tensor, float)):
                    try:
                        value = int(value)
                    except ValueError:
                        pass

            return value

        output: dict[str, object] = {"Time": datetime.utcnow().isoformat()}

        if step_nr is not None:
            output["Step"] = step_nr

        for value, descriptor in values_and_descriptors:
            output[descriptor.display_name] = sanitize(value, descriptor)

        try:
            json.dump(output, stream, indent=None)

            stream.write("\n")

            if flush:
                stream.flush()
        except OSError as ex:
            raise MetricRecordError(
                f"The metric values of the '{run}' cannot be saved to the JSON file. See the nested exception for details."
            ) from ex

    def _get_stream(self, run: str) -> TextIO:
        try:
            return self._streams[run]
        except KeyError:
            pass

        file = self._output_dir.joinpath(run).with_suffix(".jsonl")

        try:
            self._file_system.make_directory(file.parent)
        except OSError as ex:
            raise MetricRecordError(
                f"The '{file.parent}' metric directory cannot be created. See the nested exception for details."
            ) from ex

        try:
            fp = self._file_system.open_text(file, mode=FileMode.APPEND)
        except OSError as ex:
            raise MetricRecordError(
                f"The '{file}' metric file for the '{run} run cannot be created. See the nested exception for details."
            ) from ex

        self._streams[run] = fp

        return fp

    @override
    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

        self._streams.clear()


JSONL_METRIC_RECORDER: Final = "jsonl"


@dataclass(kw_only=True)
class JsonlMetricRecorderConfig:
    enabled: bool = True


@final
class JsonlMetricRecorderHandler(MetricRecorderHandler):
    _file_system: FileSystem
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(
        self, file_system: FileSystem, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        self._file_system = file_system
        self._metric_descriptors = metric_descriptors

    @override
    def create(self, output_dir: Path, config: object) -> MetricRecorder:
        config = structure(config, JsonlMetricRecorderConfig)

        validate(config)

        if not config.enabled:
            return NoopMetricRecorder()

        metrics_dir = output_dir.joinpath("metrics")

        return JsonlMetricRecorder(
            metrics_dir, self._file_system, self._metric_descriptors
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return JsonlMetricRecorderConfig

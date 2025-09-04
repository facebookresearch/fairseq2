# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO, final

from torch import Tensor
from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.metrics import format_as_int
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.recorder import MetricRecorder


@final
class JsonlMetricRecorder(MetricRecorder):
    """Records metric values to JSONL files."""

    _CATEGORY_PART_REGEX: Final = re.compile("^[-_a-zA-Z0-9]+$")

    def __init__(
        self,
        output_dir: Path,
        file_system: FileSystem,
        metric_descriptors: MetricDescriptorRegistry,
    ) -> None:
        """
        :param output_dir: The base directory under which to store the metric
            files.
        """
        self._output_dir = output_dir.joinpath("metrics")
        self._file_system = file_system
        self._metric_descriptors = metric_descriptors
        self._streams: dict[str, TextIO] = {}

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        category = category.strip()

        for part in category.split("/"):
            if re.match(self._CATEGORY_PART_REGEX, part) is None:
                raise ValueError(
                    f"`category` must contain only alphanumeric characters, dash, underscore, and forward slash, but is '{category}' instead."
                )

        stream = self._get_stream(category)

        values_and_descriptors = []

        for name, value in values.items():
            descriptor = self._metric_descriptors.maybe_get(name)
            if descriptor is None:
                descriptor = MetricDescriptor(
                    name, name, 999, formatter=lambda v: str(v)
                )

            values_and_descriptors.append((value, descriptor))

        # Sort by priority and display name.
        values_and_descriptors.sort(key=lambda p: (p[1].priority, p[1].display_name))

        def sanitize(value: object, descriptor: MetricDescriptor) -> object:
            if not isinstance(value, (int, float, str, Tensor)):
                return repr(value)

            if isinstance(value, Tensor):
                if value.numel() != 1:
                    return value.tolist()

                value = value.item()

            if descriptor.formatter is format_as_int:
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

            stream.flush()
        except OSError as ex:
            raise_operational_system_error(ex)

    def _get_stream(self, category: str) -> TextIO:
        fp = self._streams.get(category)
        if fp is not None:
            return fp

        file = self._output_dir.joinpath(category).with_suffix(".jsonl")

        try:
            self._file_system.make_directory(file.parent)
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            fp = self._file_system.open_text(file, mode=FileMode.APPEND)
        except OSError as ex:
            raise_operational_system_error(ex)

        self._streams[category] = fp

        return fp

    @override
    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

        self._streams.clear()

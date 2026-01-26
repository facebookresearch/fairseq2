# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO, final

from torch import Tensor
from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.logging import log
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
        self._remote_output_dir = output_dir.joinpath("metrics")
        self._file_system = file_system
        self._metric_descriptors = metric_descriptors
        self._streams: dict[str, TextIO] = {}
        self._local_files: dict[str, Path] = {}

        # For remote filesystems, use local temp directory for metrics
        self._is_remote = not file_system.is_local_path(output_dir)
        if self._is_remote:
            output_name = str(output_dir).replace("://", "_").replace("/", "_")[-100:]
            self._local_metrics_dir = (
                Path(tempfile.gettempdir()) / f"fairseq2_metrics_{output_name}"
            )
            self._local_metrics_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                "Metrics are buffered locally at {} (output_dir is remote)",
                self._local_metrics_dir,
            )
        else:
            self._local_metrics_dir = None

    @property
    def _output_dir(self) -> Path:
        """Get the directory to write metrics to (local for remote output dirs)."""
        if self._local_metrics_dir is not None:
            return self._local_metrics_dir
        return self._remote_output_dir

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
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

        def sanitize(value: object) -> object:
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    return value.tolist()

                value = value.item()

            if isinstance(value, (int, float, str)):
                return value

            if isinstance(value, Sequence):
                return [sanitize(e) for e in value]

            raise ValueError(
                f"`values` must consist of objects of types `{int}`, `{float}`, `{Tensor}`, and `{str}` only."
            )

        output: dict[str, object] = {"Time": datetime.utcnow().isoformat()}

        if step_nr is not None:
            output["Step"] = step_nr

        for value, descriptor in values_and_descriptors:
            output[descriptor.display_name] = sanitize(value)

        try:
            json.dump(output, stream, indent=None)

            stream.write("\n")

            stream.flush()
        except OSError as ex:
            raise_operational_system_error(ex)

    def _get_stream(self, category: str) -> TextIO:
        category = category.strip()

        fp = self._streams.get(category)
        if fp is not None:
            return fp

        for part in category.split("/"):
            if re.match(self._CATEGORY_PART_REGEX, part) is None:
                raise ValueError(
                    f"`category` must contain only alphanumeric characters, dash, underscore, and forward slash, but is '{category}' instead."
                )

        file = self._output_dir.joinpath(category).with_suffix(".jsonl")

        try:
            file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            fp = open(file, mode="a", encoding="utf-8")
        except OSError as ex:
            raise_operational_system_error(ex)

        self._streams[category] = fp
        self._local_files[category] = file

        return fp

    def _sync_to_remote(self) -> None:
        """Sync local metric files to remote storage if using remote output."""
        if not self._is_remote or self._local_metrics_dir is None:
            return

        try:
            self._file_system.make_directory(self._remote_output_dir)
        except OSError:
            pass

        for category, local_file in self._local_files.items():
            if not local_file.exists():
                continue

            remote_file = self._remote_output_dir.joinpath(category).with_suffix(
                ".jsonl"
            )
            try:
                self._file_system.make_directory(remote_file.parent)
                # Read local and write to remote
                with open(local_file, "rb") as f:
                    content = f.read()
                with self._file_system.open(remote_file, mode=FileMode.WRITE) as f:
                    f.write(content)
                log.debug("Synced metrics {} to {}", local_file, remote_file)
            except Exception as e:
                log.warning("Failed to sync metrics to remote: {}", e)

    @override
    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

        self._streams.clear()

        # Sync to remote storage on close
        self._sync_to_remote()

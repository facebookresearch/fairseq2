# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import final

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import override

from fairseq2.error import OperationalError, raise_operational_system_error
from fairseq2.file_system import get_file_system
from fairseq2.metrics.recorders.descriptor import MetricDescriptorRegistry
from fairseq2.metrics.recorders.recorder import MetricRecorder


@final
class TensorBoardRecorder(MetricRecorder):
    """Records metric values to TensorBoard."""

    def __init__(
        self, output_dir: Path, metric_descriptors: MetricDescriptorRegistry
    ) -> None:
        """
        :param output_dir:
            The base directory under which to store the TensorBoard files.
        """
        self._remote_output_dir = output_dir.joinpath("tb")
        self._metric_descriptors = metric_descriptors
        self._writers: dict[str, SummaryWriter] = {}
        self._file_system = get_file_system()

        # For remote filesystems, use a local temp directory
        if self._file_system.is_local_path(output_dir):
            self._local_output_dir = self._remote_output_dir
            self._is_remote = False
        else:
            # Create local temp directory for TensorBoard files
            self._temp_dir = tempfile.mkdtemp(prefix="fairseq2_tb_")
            self._local_output_dir = Path(self._temp_dir)
            self._is_remote = True

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        writer = self._get_writer(category)

        try:
            for name, value in values.items():
                descriptor = self._metric_descriptors.maybe_get(name)
                if descriptor is None:
                    display_name = name
                else:
                    display_name = descriptor.display_name

                self._add_value(writer, step_nr, display_name, value)

            writer.flush()
        except OSError as ex:
            raise_operational_system_error(ex)
        except RuntimeError as ex:
            raise OperationalError(
                "Metric values cannot be saved to TensorBoard."
            ) from ex

    def _add_value(
        self, writer: SummaryWriter, step_nr: int | None, name: str, value: object
    ) -> None:
        if isinstance(value, str):
            writer.add_text(name, value, step_nr)

            return

        if isinstance(value, (int, float, Tensor)):
            writer.add_scalar(name, value, step_nr)

            return

        if isinstance(value, Sequence):
            for idx, elem in enumerate(value):
                self._add_value(writer, step_nr, f"{name} ({idx})", elem)

            return

        raise ValueError(
            f"`values` must consist of objects of types `{int}`, `{float}`, `{Tensor}`, and `{str}` only."
        )

    def _get_writer(self, category: str) -> SummaryWriter:
        writer = self._writers.get(category)
        if writer is None:
            path = self._local_output_dir.joinpath(category)

            writer = SummaryWriter(path)

            self._writers[category] = writer

        return writer

    def _sync_to_remote(self) -> None:
        """Sync local TensorBoard files to remote storage."""
        if not self._is_remote:
            return

        # Copy all files from local to remote
        for local_path in self._local_output_dir.rglob("*"):
            if local_path.is_file():
                rel_path = local_path.relative_to(self._local_output_dir)
                remote_path = self._remote_output_dir.joinpath(rel_path)

                # Read local file and write to remote
                with open(local_path, "rb") as f:
                    data = f.read()

                self._file_system.make_directory(remote_path.parent)
                with self._file_system.open_for_write(remote_path) as f:
                    f.write(data)

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()

        # Sync to remote if needed
        if self._is_remote:
            try:
                self._sync_to_remote()
            finally:
                # Clean up temp directory
                shutil.rmtree(self._temp_dir, ignore_errors=True)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import final

from torch import Tensor
from typing_extensions import override

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
except ImportError:
    _has_tensorboard = False
else:
    _has_tensorboard = True

from fairseq2.error import OperationalError, raise_operational_system_error
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
        if not _has_tensorboard:
            raise OperationalError(
                "tensorboard is not found. Use `pip install tensorboard`."
            )

        self._output_dir = output_dir.joinpath("tb")
        self._metric_descriptors = metric_descriptors
        self._writers: dict[str, SummaryWriter] = {}

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

                if not isinstance(value, (int, float, str, Tensor)):
                    value = repr(value)

                writer.add_scalar(display_name, value, step_nr)

            writer.flush()
        except OSError as ex:
            raise_operational_system_error(ex)
        except RuntimeError as ex:
            raise OperationalError(
                "Metric values cannot be saved to TensorBoard."
            ) from ex

    def _get_writer(self, category: str) -> SummaryWriter:
        writer = self._writers.get(category)
        if writer is None:
            path = self._output_dir.joinpath(category)

            writer = SummaryWriter(path)

            self._writers[category] = writer

        return writer

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()

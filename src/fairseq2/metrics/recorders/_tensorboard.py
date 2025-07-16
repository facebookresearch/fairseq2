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

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
except ImportError:
    _has_tensorboard = False
else:
    _has_tensorboard = True

from fairseq2.logging import log
from fairseq2.registry import Provider
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.metrics.recorders._descriptor import MetricDescriptor
from fairseq2.metrics.recorders._handler import MetricRecorderHandler
from fairseq2.metrics.recorders._recorder import (
    MetricRecorder,
    MetricRecordError,
    NoopMetricRecorder,
)


@final
class TensorBoardRecorder(MetricRecorder):
    """Records metric values to TensorBoard."""

    _output_dir: Path
    _metric_descriptors: Provider[MetricDescriptor]
    _writers: dict[str, SummaryWriter]

    def __init__(
        self, output_dir: Path, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        """
        :param output_dir:
            The base directory under which to store the TensorBoard files.
        """
        if not _has_tensorboard:
            log.warning("tensorboard not found. Please install it with `pip install tensorboard`.")  # fmt: skip

        self._output_dir = output_dir
        self._metric_descriptors = metric_descriptors

        self._writers = {}

    @override
    def record_metric_values(
        self, section: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        writer = self._get_writer(section)
        if writer is None:
            return

        try:
            for name, value in values.items():
                try:
                    descriptor = self._metric_descriptors.get(name)
                except LookupError:
                    descriptor = None

                if descriptor is None:
                    display_name = name
                else:
                    display_name = descriptor.display_name

                writer.add_scalar(display_name, value, step_nr)

            writer.flush()
        except RuntimeError as ex:
            raise MetricRecordError(
                f"The metric values of the '{section}' section cannot be saved to TensorBoard. See the nested exception for details."
            ) from ex

    def _get_writer(self, section: str) -> SummaryWriter | None:
        if not _has_tensorboard:
            return None

        writer = self._writers.get(section)
        if writer is None:
            writer = SummaryWriter(self._output_dir.joinpath(section))

            self._writers[section] = writer

        return writer

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()


TENSORBOARD_RECORDER: Final = "tensorboard"


@dataclass(kw_only=True)
class TensorBoardRecorderConfig:
    enabled: bool = True


@final
class TensorBoardRecorderHandler(MetricRecorderHandler):
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(self, metric_descriptors: Provider[MetricDescriptor]) -> None:
        self._metric_descriptors = metric_descriptors

    @override
    def create(
        self, output_dir: Path, config: object, hyper_params: object
    ) -> MetricRecorder:
        config = structure(config, TensorBoardRecorderConfig)

        validate(config)

        if not config.enabled:
            return NoopMetricRecorder()

        tb_dir = output_dir.joinpath("tb")

        return TensorBoardRecorder(tb_dir, self._metric_descriptors)

    @property
    @override
    def name(self) -> str:
        return TENSORBOARD_RECORDER

    @property
    @override
    def config_kls(self) -> type[object]:
        return TensorBoardRecorderConfig

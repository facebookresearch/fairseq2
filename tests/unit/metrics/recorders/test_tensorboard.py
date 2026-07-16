# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fairseq2.metrics.formatters import (
    format_as_float,
    format_as_percentage,
    scale_as_percentage,
)
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.tensorboard import TensorBoardRecorder


class TestTensorBoardRecorderValueTransform:
    def setup_method(self) -> None:
        descriptors = [
            MetricDescriptor(
                "padding_ratio",
                "Padding Ratio (%)",
                835,
                format_as_percentage,
                value_transform=scale_as_percentage,
            ),
            MetricDescriptor(
                "loss",
                "Loss",
                90,
                format_as_float,
            ),
        ]
        self.registry = MetricDescriptorRegistry(descriptors)

    @patch(
        "fairseq2.metrics.recorders.tensorboard.SummaryWriter",
    )
    def test_applies_value_transform_for_percentage_metrics(
        self, mock_writer_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        recorder = TensorBoardRecorder(tmp_path, self.registry)
        recorder.record_metric_values("train", {"padding_ratio": 0.42}, step_nr=1)

        mock_writer.add_scalar.assert_called_once_with("Padding Ratio (%)", 42.0, 1)

    @patch(
        "fairseq2.metrics.recorders.tensorboard.SummaryWriter",
    )
    def test_does_not_transform_metrics_without_value_transform(
        self, mock_writer_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        recorder = TensorBoardRecorder(tmp_path, self.registry)
        recorder.record_metric_values("train", {"loss": 2.5}, step_nr=1)

        mock_writer.add_scalar.assert_called_once_with("Loss", 2.5, 1)

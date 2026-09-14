# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock

from wandb import Run as WandbRun

from fairseq2.metrics.formatters import (
    format_as_float,
    format_as_percentage,
    scale_as_percentage,
)
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.wandb import WandbRecorder


class TestWandbRecorderValueTransform:
    def setup_method(self) -> None:
        self.mock_run = MagicMock(spec=WandbRun)

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
        registry = MetricDescriptorRegistry(descriptors)
        self.recorder = WandbRecorder(self.mock_run, registry)

    def test_applies_value_transform_for_percentage_metrics(self) -> None:
        self.recorder.record_metric_values("train", {"padding_ratio": 0.42}, step_nr=1)

        self.mock_run.log.assert_called_once_with(
            {"train/Padding Ratio (%)": 42.0}, step=1
        )

    def test_does_not_transform_metrics_without_value_transform(self) -> None:
        self.recorder.record_metric_values("train", {"loss": 2.5}, step_nr=1)

        self.mock_run.log.assert_called_once_with({"train/Loss": 2.5}, step=1)

    def test_applies_transform_only_to_metrics_with_transform(self) -> None:
        self.recorder.record_metric_values(
            "train", {"padding_ratio": 0.1, "loss": 3.0}, step_nr=5
        )

        self.mock_run.log.assert_called_once()
        logged = self.mock_run.log.call_args[0][0]
        assert logged["train/Padding Ratio (%)"] == 10.0
        assert logged["train/Loss"] == 3.0

    def test_unknown_metric_passes_raw_value(self) -> None:
        self.recorder.record_metric_values("train", {"unknown_metric": 0.75}, step_nr=1)

        self.mock_run.log.assert_called_once_with(
            {"train/unknown_metric": 0.75}, step=1
        )

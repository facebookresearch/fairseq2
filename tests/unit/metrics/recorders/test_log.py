# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

import pytest

from fairseq2.metrics.formatters import (
    format_as_float,
    format_as_percentage,
    scale_as_percentage,
)
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.log import LogMetricRecorder


class TestLogRecorderValueTransform:
    """Verify LogMetricRecorder uses the formatter (not value_transform) for display."""

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
        registry = MetricDescriptorRegistry(descriptors)
        self.recorder = LogMetricRecorder(registry)

    def test_formatter_still_produces_percentage_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="fairseq2"):
            self.recorder.record_metric_values(
                "train", {"padding_ratio": 0.42}, step_nr=1
            )

        assert any("42.00%" in record.message for record in caplog.records)

    def test_formatter_still_produces_float_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="fairseq2"):
            self.recorder.record_metric_values("train", {"loss": 2.5}, step_nr=1)

        assert any("2.5" in record.message for record in caplog.records)

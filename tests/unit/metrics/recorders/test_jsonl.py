# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fairseq2.file_system import FileMode
from fairseq2.metrics.formatters import (
    format_as_float,
    format_as_percentage,
    scale_as_percentage,
)
from fairseq2.metrics.recorders.descriptor import (
    MetricDescriptor,
    MetricDescriptorRegistry,
)
from fairseq2.metrics.recorders.jsonl import JsonlMetricRecorder


def _make_mock_file_system(tmp_path: Path) -> MagicMock:
    mock_fs = MagicMock()
    mock_fs.make_directory.side_effect = lambda p: p.mkdir(parents=True, exist_ok=True)
    mock_fs.open_text.side_effect = lambda p, mode=FileMode.APPEND: open(
        p, "a" if mode == FileMode.APPEND else "r"
    )
    return mock_fs


class TestJsonlRecorderValueTransform:
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

    def test_applies_value_transform_for_percentage_metrics(
        self, tmp_path: Path
    ) -> None:
        fs = _make_mock_file_system(tmp_path)
        recorder = JsonlMetricRecorder(tmp_path, fs, self.registry)

        recorder.record_metric_values("train", {"padding_ratio": 0.42}, step_nr=1)
        recorder.close()

        jsonl_file = tmp_path / "metrics" / "train.jsonl"
        line = json.loads(jsonl_file.read_text().strip())
        assert line["Padding Ratio (%)"] == 42.0

    def test_does_not_transform_metrics_without_value_transform(
        self, tmp_path: Path
    ) -> None:
        fs = _make_mock_file_system(tmp_path)
        recorder = JsonlMetricRecorder(tmp_path, fs, self.registry)

        recorder.record_metric_values("train", {"loss": 2.5}, step_nr=1)
        recorder.close()

        jsonl_file = tmp_path / "metrics" / "train.jsonl"
        line = json.loads(jsonl_file.read_text().strip())
        assert line["Loss"] == 2.5

    def test_mixed_metrics_transforms_only_percentage(self, tmp_path: Path) -> None:
        fs = _make_mock_file_system(tmp_path)
        recorder = JsonlMetricRecorder(tmp_path, fs, self.registry)

        recorder.record_metric_values(
            "train", {"loss": 3.0, "padding_ratio": 0.1}, step_nr=5
        )
        recorder.close()

        jsonl_file = tmp_path / "metrics" / "train.jsonl"
        line = json.loads(jsonl_file.read_text().strip())
        assert line["Loss"] == 3.0
        assert line["Padding Ratio (%)"] == 10.0

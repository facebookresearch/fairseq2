# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Mapping, final

from typing_extensions import override

from fairseq2.metrics.recorders.recorder import MetricRecorder


@final
class CompositeMetricRecorder(MetricRecorder):
    _recorders: Iterable[MetricRecorder]

    def __init__(self, recorders: Iterable[MetricRecorder]) -> None:
        self._recorders = recorders

    @override
    def record_metric_values(
        self, section: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        for recorder in self._recorders:
            recorder.record_metric_values(section, values, step_nr)

    @override
    def close(self) -> None:
        for recorder in self._recorders:
            recorder.close()

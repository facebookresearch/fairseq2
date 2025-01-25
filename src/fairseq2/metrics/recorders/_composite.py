# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Mapping, Sequence, final

from typing_extensions import override

from fairseq2.metrics.recorders._recorder import MetricRecorder


@final
class CompositeMetricRecorder(MetricRecorder):
    _inner_recorders: Sequence[MetricRecorder]

    def __init__(self, recorders: Sequence[MetricRecorder]) -> None:
        self._inner_recorders = recorders

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        for recorder in self._inner_recorders:
            recorder.record_metrics(run, values, step_nr, flush=flush)

    @override
    def close(self) -> None:
        for recorder in self._inner_recorders:
            recorder.close()

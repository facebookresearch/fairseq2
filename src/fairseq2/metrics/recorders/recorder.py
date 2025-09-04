# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Final, final

from typing_extensions import override

from fairseq2.runtime.closable import Closable


class MetricRecorder(Closable):
    """Records metric values."""

    @abstractmethod
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None: ...


@final
class _NoopMetricRecorder(MetricRecorder):
    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        pass

    @override
    def close(self) -> None:
        pass


NOOP_METRIC_RECORDER: Final = _NoopMetricRecorder()

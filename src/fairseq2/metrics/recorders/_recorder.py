# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import final

from typing_extensions import override


class MetricRecorder(ABC):
    """Records metric values."""

    @abstractmethod
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        """Record ``values``.

        :param run:
            The name of the run (e.g. 'train', 'eval').
        :param values:
            The metric values.
        :param step_nr:
            The step number of the run.
        :param flush:
            If ``True``, flushes any buffers after recording.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the recorder."""


class MetricRecordError(Exception):
    pass


@final
class NoopMetricRecorder(MetricRecorder):
    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        pass

    @override
    def close(self) -> None:
        pass


def record_metrics(
    recorders: Sequence[MetricRecorder],
    run: str,
    values: Mapping[str, object],
    step_nr: int | None = None,
    *,
    flush: bool = True,
) -> None:
    """Record ``values`` to ``recorders``.

    :param recorders: The recorders to record to.
    :param run: The name of the run (e.g. 'train', 'eval').
    :param values: The metric values.
    :param step_nr: The step number of the run.
    :param flush: If ``True``, flushes any buffers after recording.
    """
    for recorder in recorders:
        recorder.record_metrics(run, values, step_nr, flush=flush)

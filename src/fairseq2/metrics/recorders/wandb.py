# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import final

from torch import Tensor
from typing_extensions import override
from wandb import Run as WandbRun

from fairseq2.error import InternalError
from fairseq2.metrics.recorders.descriptor import MetricDescriptorRegistry
from fairseq2.metrics.recorders.recorder import MetricRecorder


@final
class WandbRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    def __init__(
        self, run: WandbRun, metric_descriptors: MetricDescriptorRegistry
    ) -> None:
        self._run = run
        self._metric_descriptors = metric_descriptors

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        output: dict[str, object] = {}

        for name, value in values.items():
            descriptor = self._metric_descriptors.maybe_get(name)
            if descriptor is None:
                display_name = f"{category}/{name}"
            else:
                display_name = f"{category}/{descriptor.display_name}"

            self._add_value(display_name, value, output)

        try:
            self._run.log(output, step=step_nr)
        except (RuntimeError, ValueError, TypeError) as ex:
            raise InternalError(
                f"an unexpected error occurred while logging metric values of category '{category}' to Weights & Biases"
            ) from ex

    def _add_value(self, name: str, value: object, output: dict[str, object]) -> None:
        if isinstance(value, (int, float, Tensor, str)):
            output[name] = value

            return

        if isinstance(value, Sequence):
            for idx, elem in enumerate(value):
                self._add_value(f"{name} ({idx})", elem, output)

            return

        raise ValueError(
            "`values` must consist of objects of types `int`, `float`, `Tensor`, and `str` only"
        )

    @override
    def close(self) -> None:
        self._run.finish()

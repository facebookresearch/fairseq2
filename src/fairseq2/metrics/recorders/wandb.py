# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final

from typing_extensions import override

from fairseq2.error import InfraError
from fairseq2.metrics.recorders.descriptor import MetricDescriptor
from fairseq2.metrics.recorders.recorder import MetricRecorder
from fairseq2.runtime.provider import Provider


@final
class WandbRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    _run: Any
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(
        self, run: Any, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        """
        In order to use W&B, run `wandb login` from the command line and enter
        the API key when prompted.
        """
        self._run = run

        self._metric_descriptors = metric_descriptors

    @override
    def record_metric_values(
        self, section: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        for name, value in values.items():
            try:
                descriptor = self._metric_descriptors.get(name)
            except LookupError:
                descriptor = None

            if descriptor is None:
                display_name = f"{section}/{name}"
            else:
                display_name = f"{section}/{descriptor.display_name}"

            try:
                self._run.log({display_name: value}, step=step_nr)
            except RuntimeError as ex:
                raise InfraError(
                    f"The metric values of the '{section}' section cannot be saved to Weights & Biases. See the nested exception for details."
                ) from ex

    @override
    def close(self) -> None:
        self._run.finish()

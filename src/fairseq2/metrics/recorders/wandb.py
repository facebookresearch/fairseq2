# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final

from typing_extensions import override

from fairseq2.error import OperationalError
from fairseq2.metrics.recorders.descriptor import MetricDescriptorRegistry
from fairseq2.metrics.recorders.recorder import MetricRecorder


class WandbClient:
    def __init__(self, run: Any) -> None:
        self.run = run


@final
class WandbRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    def __init__(
        self, client: WandbClient, metric_descriptors: MetricDescriptorRegistry
    ) -> None:
        """
        In order to use W&B, run `wandb login` from the command line and enter
        the API key when prompted.
        """
        self._run = client.run
        self._metric_descriptors = metric_descriptors

    @override
    def record_metric_values(
        self, category: str, values: Mapping[str, object], step_nr: int | None = None
    ) -> None:
        for name, value in values.items():
            descriptor = self._metric_descriptors.maybe_get(name)
            if descriptor is None:
                display_name = f"{category}/{name}"
            else:
                display_name = f"{category}/{descriptor.display_name}"

            try:
                self._run.log({display_name: value}, step=step_nr)
            except RuntimeError as ex:
                raise OperationalError(
                    "Metric values cannot be saved to Weights & Biases."
                ) from ex

    @override
    def close(self) -> None:
        self._run.finish()

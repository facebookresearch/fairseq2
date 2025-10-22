# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, final

from torch import Tensor
from typing_extensions import override

from fairseq2.error import OperationalError, raise_operational_system_error
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
        """
        Retrieves and stores a `descriptor` and `value` as a ``Mapping`` for each metric
        :raises OSError: If an operational system error occurs (file not found, permission issue, connection problem)
        :raises RuntimeError: If 
        """
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
        except OSError as ex:
            raise_operational_system_error(ex)
        except RuntimeError as ex:
            raise OperationalError(
                "Metric values cannot be saved to Weights & Biases."
            ) from ex

    def _add_value(self, name: str, value: object, output: dict[str, object]) -> None:
        """
        Adds a value to the output dictionary
        :raises ValueError: If `values` are not of type int, float, Tensor or str
        """
        if isinstance(value, (int, float, Tensor, str)):
            output[name] = value

            return

        if isinstance(value, Sequence):
            for idx, elem in enumerate(value):
                self._add_value(f"{name} ({idx})", elem, output)

            return

        raise ValueError(
            f"`values` must consist of objects of types `{int}`, `{float}`, `{Tensor}`, and `{str}` only."
        )

    @override
    def close(self) -> None:
        """
        Close wandb logger and end run
        """
        self._run.finish()

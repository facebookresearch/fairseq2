# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.logging import log
from fairseq2.metrics import MetricDescriptor
from fairseq2.metrics.recorders._handler import MetricRecorderHandler
from fairseq2.metrics.recorders._recorder import (
    MetricRecorder,
    MetricRecordError,
    NoopMetricRecorder,
)
from fairseq2.registry import Provider
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import ValidationError, ValidationResult, validate

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    has_wandb = False
else:
    has_wandb = True


@final
class WandbRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(
        self,
        project: str,
        name: str,
        output_dir: Path,
        metric_descriptors: Provider[MetricDescriptor],
    ) -> None:
        """
        :param project: The W&B project name.
        :param name: The run name.
        :param output_dir: The base directory under which to store the W&B files.

        In order to use W&B, run `wandb login` from the command line and enter
        the API key when prompted.
        """
        if not has_wandb:
            log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

            self._run = None
        else:
            self._run = wandb.init(
                project=project, name=name, dir=output_dir.parent, resume="allow"
            )

        self._metric_descriptors = metric_descriptors

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, object],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        if self._run is None:
            return

        for name, value in values.items():
            try:
                descriptor = self._metric_descriptors.get(name)
            except LookupError:
                descriptor = None

            if descriptor is None:
                display_name = name
            else:
                display_name = descriptor.display_name

            try:
                self._run.log({display_name: value}, step=step_nr)
            except RuntimeError as ex:
                raise MetricRecordError(
                    f"The metric values of the '{run}' cannot be saved to Weights & Biases. See the nested exception for details."
                ) from ex

    @override
    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


WANDB_RECORDER: Final = "wandb"


@dataclass(kw_only=True)
class WandbRecorderConfig:
    enabled: bool = False

    project: str | None = None

    run: str | None = None

    def validate(self) -> None:
        result = ValidationResult()

        if self.enabled:
            if self.project is None or self.run is None:
                result.add_error(
                    "Both `project` and `run` must be specified when `enabled` is set."
                )

        if result.has_error:
            raise ValidationError(
                "The Weights & Biases recorder configuration has one or more validation errors:", result  # fmt: skip
            )


@final
class WandbRecorderHandler(MetricRecorderHandler):
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(self, metric_descriptors: Provider[MetricDescriptor]) -> None:
        self._metric_descriptors = metric_descriptors

    @override
    def create(self, output_dir: Path, config: object) -> MetricRecorder:
        config = structure(config, WandbRecorderConfig)

        validate(config)

        if not config.enabled:
            return NoopMetricRecorder()

        if config.project is None or config.run is None:
            raise ValueError(
                "`config.project` and `config.run` must be specified when `config.enabled` is set."
            )

        wandb_dir = output_dir.joinpath("wandb")

        return WandbRecorder(
            config.project, config.run, wandb_dir, self._metric_descriptors
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return WandbRecorderConfig

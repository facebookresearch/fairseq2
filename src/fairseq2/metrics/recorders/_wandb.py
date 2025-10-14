# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, final

from typing_extensions import override

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

from fairseq2.file_system import FileMode, FileSystem
from fairseq2.logging import log
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError, structure, unstructure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.metrics.recorders._descriptor import MetricDescriptor
from fairseq2.metrics.recorders._handler import MetricRecorderHandler
from fairseq2.metrics.recorders._recorder import (
    MetricRecorder,
    MetricRecordError,
    NoopMetricRecorder,
)


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
                raise MetricRecordError(
                    f"The metric values of the '{section}' section cannot be saved to Weights & Biases. See the nested exception for details."
                ) from ex

    @override
    def close(self) -> None:
        self._run.finish()


WANDB_RECORDER: Final = "wandb"


WandbResumeMode: TypeAlias = Literal["allow", "never", "must", "auto"]


@dataclass(kw_only=True)
class WandbRecorderConfig:
    enabled: bool = False

    entity: str | None = None

    project: str | None = None

    run_id: str | None = "auto"

    run_name: str | None = None

    group: str | None = None

    job_type: str | None = None

    resume_mode: WandbResumeMode = "allow"


@final
class WandbRecorderHandler(MetricRecorderHandler):
    _file_system: FileSystem
    _metric_descriptors: Provider[MetricDescriptor]

    def __init__(
        self, file_system: FileSystem, metric_descriptors: Provider[MetricDescriptor]
    ) -> None:
        self._file_system = file_system
        self._metric_descriptors = metric_descriptors

    @override
    def create(
        self, output_dir: Path, config: object, hyper_params: object
    ) -> MetricRecorder:
        config = structure(config, WandbRecorderConfig)

        validate(config)

        if not config.enabled:
            return NoopMetricRecorder()

        if not _has_wandb:
            log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

            return NoopMetricRecorder()

        if hyper_params is not None:
            try:
                hyper_params = unstructure(hyper_params)
            except StructureError as ex:
                raise ValueError(
                    "`hyper_params` cannot be unstructured. See the nested exception for details."
                ) from ex

            if not isinstance(hyper_params, dict):
                raise TypeError(
                    f"The unstructured form of `hyper_params` must be of type `dict`, but is of type `{type(hyper_params)}` instead."
                )

        run_id = self._get_run_id(output_dir, config)

        try:
            run = wandb.init(
                entity=config.entity,
                project=config.project,
                dir=output_dir,
                id=run_id,
                name=config.run_name,
                config=hyper_params,
                group=config.group,
                job_type=config.job_type,
                resume=config.resume_mode,
                settings=wandb.Settings(init_timeout=120),
            )
        except (RuntimeError, ValueError) as ex:
            raise MetricRecordError(
                "Weights & Biases client cannot be initialized. See the nested exception for details."
            ) from ex

        return WandbRecorder(run, self._metric_descriptors)

    def _get_run_id(self, output_dir: Path, config: WandbRecorderConfig) -> str:
        run_id = config.run_id

        if run_id is None:
            return wandb.util.generate_id()

        if run_id != "auto":
            return run_id

        wandb_file = output_dir.joinpath("wandb_run_id")

        try:
            fp = self._file_system.open_text(wandb_file)

            with fp:
                return fp.read()
        except FileNotFoundError:
            pass
        except OSError as ex:
            raise MetricRecordError(
                "The Weights & Biases run ID cannot be loaded. See the nested exception for details."
            ) from ex

        run_id = wandb.util.generate_id()

        try:
            fp = self._file_system.open_text(wandb_file, mode=FileMode.WRITE)

            with fp:
                fp.write(run_id)
        except OSError as ex:
            raise MetricRecordError(
                "The Weights & Biases run ID cannot be saved. See the nested exception for details."
            ) from ex

        return run_id

    @property
    @override
    def name(self) -> str:
        return WANDB_RECORDER

    @property
    @override
    def config_kls(self) -> type[object]:
        return WandbRecorderConfig

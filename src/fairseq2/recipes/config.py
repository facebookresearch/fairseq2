# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import torch

from fairseq2.generation import (
    BEAM_SEARCH_GENERATOR,
    SAMPLING_GENERATOR,
    BeamSearchConfig,
    SamplingConfig,
)
from fairseq2.metrics.recorders import (
    JSONL_METRIC_RECORDER,
    LOG_METRIC_RECORDER,
    TENSORBOARD_RECORDER,
    WANDB_RECORDER,
    JsonlMetricRecorderConfig,
    LogMetricRecorderConfig,
    TensorBoardRecorderConfig,
    WandbRecorderConfig,
)
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.profilers import TORCH_PROFILER, TorchProfilerConfig
from fairseq2.typing import DataType


@dataclass(kw_only=True)
class ModelSection:
    name: str | None = None

    family: str | None = None

    arch: str | None = None

    config: object = None

    checkpoint: Path | None = None


@dataclass
class ReferenceModelSection:
    name: str


@dataclass(kw_only=True)
class DatasetSection:
    name: str | None

    path: Path | None = None

    family: str


@dataclass(kw_only=True)
class TextTokenizerSection:
    name: str


@dataclass(kw_only=True)
class GangSection:
    tensor_parallel_size: int = 1

    timeout: int = 15

    high_priority: bool = True

    monitored: bool = False


DataParallelism: TypeAlias = Literal["ddp", "fsdp"]


@dataclass(kw_only=True)
class TrainerSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    data_parallelism: DataParallelism = "ddp"
    """The data parallelism API to use."""

    fsdp: FsdpSection = field(default_factory=lambda: FsdpSection())

    mixed_precision: Literal["static", "dynamic"] | None = "static"
    """
    If 'none', the whole training will be run in `dtype`. If 'static', forward
    and backward passes will be run in `dtype`, but the optimizer step will be
    run in full precision. If 'dynamic', forward and backward passes will be run
    with `torch.amp` in `dtype`, but the optimizer step will be run in full
    precision.
    """

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    activation_checkpointing: bool = False
    """If ``True``, uses layer-wise activation checkpointing."""

    max_gradient_norm: float | None = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    torch_compile: bool = False

    profile: tuple[int, int] | None = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    gradient_check: bool = False
    """if ``True``, ensures that gradients are in sync across processes."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""


FsdpGranularity: TypeAlias = Literal["layer", "stack", "model"]


@dataclass(kw_only=True)
class FsdpSection:
    version: Literal["v1", "v2"] = "v1"
    """The PyTorch FSDP version."""

    granularity: FsdpGranularity = "layer"
    """The granularity at which to wrap the model."""

    hsdp: bool = False
    """If ``True``, uses hybrid sharded data parallelism."""

    reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    fp32_reduce: bool = False


@dataclass(kw_only=True)
class OptimizerSection:
    name: str = ADAMW_OPTIMIZER

    config: object = field(default_factory=AdamWConfig)


@dataclass(kw_only=True)
class LRSchedulerSection:
    name: str | None = None

    config: object = None


@dataclass(kw_only=True)
class RegimeSection:
    num_steps: int | None = None
    """The maximum number of steps to train for. Note that num_steps is used as CosineLRScheduler argument!"""

    num_data_epochs: int | None = None
    """The maximum number of data epochs to train for."""

    score_metric: str | None = None

    lower_score_better: bool = False

    validate_after_n_steps: int = 0
    """The number of steps after which to start validating the model."""

    validate_every_n_steps: int | None = None
    """The step interval at which to validate the model."""

    validate_after_n_data_epochs: int = 0

    validate_every_n_data_epochs: int | None = None

    checkpoint_after_n_steps: int = 0

    checkpoint_every_n_steps: int | None = None
    """The step interval at which to checkpoint."""

    checkpoint_after_n_data_epochs: int = 0

    checkpoint_every_n_data_epochs: int | None = None
    """The data epoch interval at which to checkpoint."""

    keep_last_n_checkpoints: int | None = None
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    keep_best_n_checkpoints: int | None = None

    keep_last_n_models: int | None = None
    """The number of checkpoint models to keep. If ``None``, none will be deleted."""

    keep_best_n_models: int | None = None

    publish_metrics_after_n_steps: int = 0

    publish_metrics_every_n_steps: int | None = None
    """The step interval at which to publish training metrics."""

    publish_metrics_after_n_data_epochs: int = 0

    publish_metrics_every_n_data_epochs: int | None = None
    """The data epoch interval at which to publish training metrics."""


@dataclass(kw_only=True)
class EvaluatorSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    torch_compile: bool = False


@dataclass(kw_only=True)
class GeneratorSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    torch_compile: bool = False


@dataclass(kw_only=True)
class CommonSection:
    metric_recorders: dict[str, object] = field(
        default_factory=lambda: {
            LOG_METRIC_RECORDER: LogMetricRecorderConfig(),
            JSONL_METRIC_RECORDER: JsonlMetricRecorderConfig(),
            TENSORBOARD_RECORDER: TensorBoardRecorderConfig(),
            WANDB_RECORDER: WandbRecorderConfig(),
        }
    )

    profilers: dict[str, object] = field(
        default_factory=lambda: {
            TORCH_PROFILER: TorchProfilerConfig(),
        }
    )

    assets: AssetsSection = field(default_factory=lambda: AssetsSection())

    seed: int = 2


@dataclass(kw_only=True)
class AssetsSection:
    extra_path: Path | None = None

    checkpoint_dir: Path | None = None
    """If not ``None``, adds the specified path to the default asset store."""


@dataclass(kw_only=True)
class SequenceGeneratorSection:
    name: str = SAMPLING_GENERATOR

    config: object = field(default_factory=SamplingConfig)

    batch_size: int = 1


@dataclass(kw_only=True)
class Seq2SeqGeneratorSection:
    name: str = BEAM_SEARCH_GENERATOR

    config: object = field(default_factory=BeamSearchConfig)

    batch_size: int = 1


ConfigSectionT = TypeVar("ConfigSectionT")


def get_config_section(
    config: object, name: str, kls: type[ConfigSectionT]
) -> ConfigSectionT:
    try:
        section = getattr(config, name)
    except AttributeError:
        raise ConfigSectionNotFoundError(name) from None

    if not isinstance(section, kls):
        raise TypeError(
            f"The '{name}' configuration section must be of type `{kls}`, but is of type `{type(section)}` instead."
        )

    return section


class ConfigSectionNotFoundError(Exception):
    section: str

    def __init__(self, section: str) -> None:
        super().__init__(
            f"The recipe configuration does not have a section named '{section}'."
        )

        self.section = section

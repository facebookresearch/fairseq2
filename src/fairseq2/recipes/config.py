# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from fairseq2.data_type import DataType
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
from fairseq2.nn.data_parallel import FSDPGranularity
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.profilers import TORCH_PROFILER, TorchProfilerConfig
from fairseq2.utils.validation import ValidationError, ValidationResult


@dataclass(kw_only=True)
class ModelSection:
    name: str | None = None

    family: str | None = None

    arch: str | None = None

    config_overrides: object = None

    path: Path | None = None

    mmap: bool = False

    compile: bool = False

    compile_options: CompileOptionsSection = field(
        default_factory=lambda: CompileOptionsSection()
    )

    def validate(self) -> None:
        result = ValidationResult()

        if self.path is not None:
            if self.family is None:
                result.add_error("`family` must be specified when `path` is specified.")
        elif self.name is None and self.family is None:
            result.add_error("Either `name` or `family` must be specified.")

        if result.has_error:
            raise ValidationError(
                "The model configuration section has one or more validation errors:", result  # fmt: skip
            )


@dataclass
class ReferenceModelSection:
    name: str

    mmap: bool = False

    compile: bool = False

    compile_options: CompileOptionsSection = field(
        default_factory=lambda: CompileOptionsSection()
    )


CompilationMode: TypeAlias = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]


@dataclass(kw_only=True)
class CompileOptionsSection:
    fullgraph: bool = False

    dynamic: bool | None = None

    mode: CompilationMode = "default"

    backend: str = "inductor"

    backend_options: dict[str, object] | None = None


@dataclass(kw_only=True)
class DatasetSection:
    name: str | None

    path: Path | None = None

    family: str

    def validate(self) -> None:
        result = ValidationResult()

        if self.name is None and self.path is None:
            result.add_error("Either `name` or `path` must be specified.")

        if result.has_error:
            raise ValidationError(
                "The dataset configuration section has one or more validation errors:", result  # fmt: skip
            )


@dataclass(kw_only=True)
class TextTokenizerSection:
    name: str


@dataclass(kw_only=True)
class GangSection:
    tensor_parallel_size: int = 1

    timeout: int = 15

    high_priority: bool = True


@dataclass(kw_only=True)
class TrainerSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp: FSDPSection = field(default_factory=lambda: FSDPSection())

    mixed_precision: Literal["static", "dynamic", "off"] = "static"
    """
    If 'none', the whole training will be run in `dtype`. If 'static', forward
    and backward passes will be run in `dtype`, but the optimizer step will be
    run in full precision. If 'dynamic', forward and backward passes will be run
    with `torch.amp` in `dtype`, but the optimizer step will be run in full
    precision.
    """

    grad_accumulation: GradAccumulationSection = field(
        default_factory=lambda: GradAccumulationSection()
    )

    activation_checkpointing: ActivationCheckpointingSection = field(
        default_factory=lambda: ActivationCheckpointingSection()
    )

    max_grad_norm: float | None = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (16, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gc_every_n_steps: int | None = None
    """If specified, calls CPython's ``gc.collect()`` every N steps."""

    grad_check: bool = False
    """If ``True``, ensures that gradients are in sync across processes."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""

    def validate(self) -> None:
        result = ValidationResult()

        if self.grad_accumulation.num_batches <= 0:
            result.add_error(
                "`grad_accumulation.num_batches` must be greater than or equal to 1."
            )

        if self.activation_checkpointing.every_nth_layer <= 0:
            result.add_error(
                "`activation_checkpointing.every_nth_layer` must be greater than or equal to 1."
            )

        if self.gc_every_n_steps is not None:
            if self.gc_every_n_steps <= 0:
                result.add_error(
                    "`gc_every_n_steps must be greater than or equal to 1."
                )

        if result.has_error:
            raise ValidationError(
                "The trainer configuration section has one or more validation errors:", result  # fmt: skip
            )


@dataclass(kw_only=True)
class FSDPSection:
    version: Literal["v1", "v2"] = "v1"
    """The PyTorch FSDP version."""

    granularity: FSDPGranularity = "layer"
    """The granularity at which to wrap the model."""

    hybrid: bool = False
    """If ``True``, uses hybrid sharded data parallelism."""

    reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    fp32_reduce: bool = False


@dataclass(kw_only=True)
class GradAccumulationSection:
    num_batches: int = 1
    """The number of batches to accumulate gradients before an optimizer update."""

    no_sync: bool = False


@dataclass(kw_only=True)
class ActivationCheckpointingSection:
    mode: Literal["off", "layerwise"] = "off"

    every_nth_layer: int = 1


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
    """The maximum number of steps to train for."""

    num_data_epochs: int | None = None
    """The maximum number of data epochs to train for."""

    validate_at_start: bool = False
    """If ``True``, runs validation before starting training."""

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

    save_model_only: bool = False

    save_as_hugging_face: bool = False

    keep_last_n_checkpoints: int | None = None
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    keep_best_n_checkpoints: int | None = None

    keep_checkpoint_every_n_steps: int | None = None

    publish_metrics_after_n_steps: int = 0

    publish_metrics_every_n_steps: int | None = None
    """The step interval at which to publish training metrics."""

    publish_metrics_after_n_data_epochs: int = 0

    publish_metrics_every_n_data_epochs: int | None = None
    """The data epoch interval at which to publish training metrics."""

    def validate(self) -> None:
        result = ValidationResult()

        if self.num_steps is not None:
            if self.num_steps <= 0:
                result.add_error("`num_steps` must be greater than or equal to 1.")

        if self.num_data_epochs is not None:
            if self.num_data_epochs <= 0:
                result.add_error(
                    "`num_data_epochs` must be greater than or equal to 1."
                )

        if self.validate_every_n_steps is not None:
            if self.validate_every_n_steps <= 0:
                result.add_error(
                    "`validate_every_n_steps` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_steps is not None:
                if self.validate_every_n_steps % self.publish_metrics_every_n_steps != 0:  # fmt: skip
                    result.add_error(
                        f"`validate_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({self.publish_metrics_every_n_steps}), but is {self.validate_every_n_steps} instead."
                    )

        if self.validate_every_n_data_epochs is not None:
            if self.validate_every_n_data_epochs <= 0:
                result.add_error(
                    "`validate_every_n_data_epochs` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_data_epochs is not None:
                if self.validate_every_n_data_epochs % self.publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    result.add_error(
                        f"`validate_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({self.publish_metrics_every_n_data_epochs}), but is {self.validate_every_n_data_epochs} instead."
                    )

        if self.checkpoint_every_n_steps is not None:
            if self.checkpoint_every_n_steps <= 0:
                result.add_error(
                    "`checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_steps is not None:
                if self.checkpoint_every_n_steps % self.publish_metrics_every_n_steps != 0:  # fmt: skip
                    result.add_error(
                        f"`checkpoint_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({self.publish_metrics_every_n_steps}), but is {self.checkpoint_every_n_steps} instead."
                    )

        if self.checkpoint_every_n_data_epochs is not None:
            if self.checkpoint_every_n_data_epochs <= 0:
                result.add_error(
                    "`checkpoint_every_n_data_epochs` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_data_epochs is not None:
                if self.checkpoint_every_n_data_epochs % self.publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    result.add_error(
                        f"`checkpoint_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({self.publish_metrics_every_n_data_epochs}), but is {self.checkpoint_every_n_data_epochs} instead."
                    )

        if self.keep_last_n_checkpoints is not None:
            if self.keep_last_n_checkpoints <= 0:
                result.add_error(
                    "`keep_last_n_checkpoints` must be greater than or equal to 1."
                )

        if self.keep_best_n_checkpoints is not None:
            if self.keep_best_n_checkpoints <= 0:
                result.add_error(
                    "`keep_best_n_checkpoints` must be greater than or equal to 1."
                )

            if self.checkpoint_every_n_steps is not None:
                if self.validate_every_n_steps is None:
                    result.add_error(
                        "`validate_every_n_steps` must be specified when `keep_best_n_checkpoints` and `checkpoint_every_n_steps` are specified."
                    )
                elif self.checkpoint_every_n_steps % self.validate_every_n_steps != 0:
                    result.add_error(
                        f"`checkpoint_every_n_steps` must be a multiple of `validate_every_n_steps` ({self.validate_every_n_steps}), but is {self.checkpoint_every_n_steps} instead."
                    )

        if self.keep_checkpoint_every_n_steps is not None:
            if self.keep_checkpoint_every_n_steps <= 0:
                result.add_error(
                    "`keep_checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if self.checkpoint_every_n_steps is not None:
                if (
                    self.keep_checkpoint_every_n_steps % self.checkpoint_every_n_steps
                    != 0
                ):
                    result.add_error(
                        f"`keep_checkpoint_every_n_steps` must be a multiple of `checkpoint_every_n_steps` ({self.checkpoint_every_n_steps}), but is {self.keep_checkpoint_every_n_steps} instead."
                    )

        if self.publish_metrics_every_n_steps is not None:
            if self.publish_metrics_every_n_steps <= 0:
                result.add_error(
                    "`publish_metrics_every_n_steps` must be greater than or equal to 1."
                )

        if self.publish_metrics_every_n_data_epochs is not None:
            if self.publish_metrics_every_n_data_epochs <= 0:
                result.add_error(
                    "`publish_metrics_every_n_data_epochs` must be greater than or equal to 1."
                )

        if result.has_error:
            raise ValidationError(
                "The regime configuration section has one or more validation errors:", result  # fmt: skip
            )


@dataclass(kw_only=True)
class EvaluatorSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""


@dataclass(kw_only=True)
class GeneratorSection:
    dtype: DataType = torch.float32
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""


@dataclass(kw_only=True)
class CommonSection:
    torch: TorchSection = field(default_factory=lambda: TorchSection())

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


SDPAVariant: TypeAlias = Literal[
    "torch",
    "torch_math",
    "torch_mem_efficient",
    "torch_flash",
    "flash2",
    "flash3",
    "naive",
]


@dataclass(kw_only=True)
class TorchSection:
    num_threads: int | None = None
    """
    The number of threads to use for intra-op parallelism in PyTorch. If ``None``,
    and the ``OMP_NUM_THREADS`` environment variable is not set, it will be set
    to the number of CPU cores divided by the local world size.
    """

    allow_tf32: bool = True
    """If ``True``, allows PyTorch to use TensorFloat32 tensor cores."""

    fp16_reduced_precision: bool = True
    """If ``True``, fp16 GEMMs are done with reduced precision reductions."""

    bf16_reduced_precision: bool = True
    """If ``True``, bf16 GEMMs are done with reduced precision reductions."""

    default_sdpa: SDPAVariant = "torch"
    """The default scaled dot-product attention variant."""

    compiled_region_activation_memory_budget: float = 1.0
    """
    The knob to adjust the activation memory budget of compiled regions. Lower
    values reduce the memory budget by recomputing activations during backward
    pass (experimental).
    """


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

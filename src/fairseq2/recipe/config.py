# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypeVar, final

import torch
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import BeamSearchAlgorithm
from fairseq2.generation.sampling import Sampler
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.error import (
    BeamSearchAlgorithmNotKnownError,
    LRSchedulerNotKnownError,
    OptimizerNotKnownError,
    SamplerNotKnownError,
    SequenceGeneratorNotKnownError,
)
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.structured import StructureError
from fairseq2.utils.validation import Validatable, ValidationResult

ConfigT = TypeVar("ConfigT")


@final
class RecipeConfig:
    def __init__(self, inner_config: object) -> None:
        self._inner_config = inner_config

    def as_(self, kls: type[ConfigT]) -> ConfigT:
        if not isinstance(self._inner_config, kls):
            raise TypeError(
                f"Recipe configuration is expected to be of type `{kls}`, but is of type `{type(self._inner_config)}` instead."
            )

        return self._inner_config


class SupportsStructure(ABC):
    @abstractmethod
    def structure(self, resolver: DependencyResolver) -> None: ...


Default: TypeAlias = Literal["default"]
"""
Indicates that the default value of a particular configuration field, as defined
in a higher-level configuration, can be used. See :class:`AdamWGroupConfig`
as an example.
"""

default: Final = "default"
"""
The singleton sentinel value assignable to a field of type ``Default``
indicating that the default value must be used.

.. code:: python

    from dataclasses import dataclass, field

    from fairseq2.recipe.config import Default, default

    @dataclass(kw_only=True)
    class MyOptimizerConfig:
        lr: float = 0.01
        \"\"\"The top-level, default learning rate.\"\"\"

        groups: list[MyOptimizerGroupConfig] = field(default_factory=list)
        \"\"\"The parameter groups.\"\"\"

    @dataclass(kw_only=True)
    class MyOptimizerGroupConfig:
        lr: float | Default = default
        \"\"\"
        The learning rate of the parameter group. If ``default``, uses 0.01 as
        specified in ``MyOptimizerConfig``.
        \"\"\"
"""


@dataclass(kw_only=True)
class ModelSection(Validatable):
    name: str | None = None

    path: Path | None = None

    family: str | None = None

    arch: str | None = None

    config_overrides: object | None = None

    dtype: DataType = torch.float32

    mmap: bool = False

    compile: bool = False

    compile_options: CompileOptions = field(default_factory=lambda: CompileOptions())

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.path is not None:
            if self.family is None:
                result.add_error("`family` must be specified when `path` is specified.")
        elif self.name is None and self.family is None:
            result.add_error("`name` or `family` must be specified.")

        return result


@dataclass
class ReferenceModelSection(Validatable):
    name: str | None = None

    path: Path | None = None

    family: str | None = None

    arch: str | None = None

    config_overrides: object | None = None

    dtype: DataType = torch.float32

    mmap: bool = False

    compile: bool = False

    compile_options: CompileOptions = field(default_factory=lambda: CompileOptions())

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.path is not None:
            if self.family is None:
                result.add_error("`family` must be specified when `path` is specified.")
        elif self.name is None:
            result.add_error("`name` or `path` must be specified.")

        return result


CompilationMode: TypeAlias = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]


@dataclass(kw_only=True)
class CompileOptions:
    fullgraph: bool = False

    dynamic: bool | None = None

    mode: CompilationMode = "default"

    backend: str = "inductor"

    backend_options: dict[str, object] | None = None


@dataclass(kw_only=True)
class DatasetSection(Validatable):
    name: str | None = None

    family: str | None = None

    config_overrides: object | None = None

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.name is None and self.family is None:
            result.add_error("`name` or `family` must be specified.")

        return result


@dataclass(kw_only=True)
class TokenizerSection(Validatable):
    name: str | None = None

    path: Path | None = None

    family: str | None = None

    config_overrides: object | None = None

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.path is not None:
            if self.family is None:
                result.add_error("`family` must be specified when `path` is specified.")
        elif self.name is None:
            result.add_error("`name` or `path` must be specified.")

        return result


@dataclass(kw_only=True)
class GangSection(Validatable):
    tensor_parallel_size: int = 1

    timeout: int = 15

    high_priority: bool = True

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.tensor_parallel_size < 1:
            result.add_error(
                "`tensor_parallel_size` must be greater than or equal to 1."
            )

        if self.timeout < 1:
            result.add_error("`timeout` must be greater than or equal to 1.")

        return result


@dataclass(kw_only=True)
class TrainerSection(Validatable):
    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp: FSDPConfig = field(default_factory=lambda: FSDPConfig())

    mixed_precision: MixedPrecisionConfig = field(
        default_factory=lambda: MixedPrecisionConfig()
    )

    grad_accumulation: GradAccumulationConfig = field(
        default_factory=lambda: GradAccumulationConfig()
    )

    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=lambda: ActivationCheckpointingConfig()
    )

    max_grad_norm: float | None = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gc_every_n_steps: int | None = 1000
    """If specified, calls CPython's ``gc.collect()`` every N steps."""

    grad_check: bool = False
    """If ``True``, ensures that gradients are in sync across processes."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.gc_every_n_steps is not None:
            if self.gc_every_n_steps < 1:
                result.add_error(
                    "`gc_every_n_steps` must be greater than or equal to 1."
                )

        return result


FSDPGranularity: TypeAlias = Literal["model", "stack", "layer"]


@dataclass(kw_only=True)
class FSDPConfig:
    version: Literal["v1", "v2"] = "v2"
    """The PyTorch FSDP version."""

    granularity: FSDPGranularity = "layer"
    """
    The granularity at which to wrap the model. Common values are layer, stack,
    and model.
    """

    hybrid: bool = False
    """If ``True``, uses hybrid sharded data parallelism."""

    reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    fp32_reduce: bool = True


@dataclass(kw_only=True)
class MixedPrecisionConfig:
    mode: Literal["off", "static", "auto"] = "static"
    """
    If 'off', the whole training will be run in `model.dtype`. If 'static',
    forward and backward passes will be run in `dtype`, but the optimizer step
    will be run in full precision. If 'auto', forward and backward passes will
    be run with `torch.amp` in `dtype`, but the optimizer step will be run in
    full precision.
    """

    dtype: DataType = torch.bfloat16


@dataclass(kw_only=True)
class GradAccumulationConfig(Validatable):
    num_batches: int = 1
    """The number of batches to accumulate gradients before an optimizer update."""

    no_sync: bool = False

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.num_batches < 1:
            result.add_error("`num_batches` must be greater than or equal to 1.")

        return result


@dataclass(kw_only=True)
class ActivationCheckpointingConfig(Validatable):
    mode: Literal["off", "layerwise"] = "off"

    every_nth_layer: int = 1

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.every_nth_layer < 1:
            result.add_error("`every_nth_layer` must be greater than or equal to 1.")

        return result


@dataclass(kw_only=True)
class RegimeSection(Validatable):
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

    score_metric: str | None = None

    checkpoint_after_n_steps: int = 0

    checkpoint_every_n_steps: int | None = None
    """The step interval at which to checkpoint."""

    checkpoint_after_n_data_epochs: int = 0

    checkpoint_every_n_data_epochs: int | None = None
    """The data epoch interval at which to checkpoint."""

    save_model_only: bool | Literal["all", "all_but_last"] = False
    """
    If ``False``, the full state of the training job is saved, including the
    trainer, model, optimizer, and data reader states. This is the conventional
    checkpointing behavior.

    If ``True`` or 'all', only the model state is saved during checkpointing.
    This is beneficial for short-lived training jobs where the user does not
    expect to resume the job but requires frequent snapshots of the model for
    evaluation purposes. In this mode, checkpointing is faster and disk space is
    saved by avoiding the storage of trainer, optimizer, and data reader states.

    If 'all_but_last', only the last checkpoint will have its full state saved,
    while all previous checkpoints will store only the model state. This is
    helpful to avoid unnecessary disk space use if the user does not plan to
    branch off the training from a previous checkpoint.
    """

    export_hugging_face: bool = False

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

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.num_steps is not None:
            if self.num_steps < 1:
                result.add_error("`num_steps` must be greater than or equal to 1.")

        if self.num_data_epochs is not None:
            if self.num_data_epochs < 1:
                result.add_error(
                    "`num_data_epochs` must be greater than or equal to 1."
                )

        if self.validate_every_n_steps is not None:
            if self.validate_every_n_steps < 1:
                result.add_error(
                    "`validate_every_n_steps` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_steps is not None:
                if self.validate_every_n_steps % self.publish_metrics_every_n_steps != 0:  # fmt: skip
                    result.add_error(
                        f"`validate_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({self.publish_metrics_every_n_steps}), but is {self.validate_every_n_steps} instead."
                    )

        if self.validate_every_n_data_epochs is not None:
            if self.validate_every_n_data_epochs < 1:
                result.add_error(
                    "`validate_every_n_data_epochs` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_data_epochs is not None:
                if self.validate_every_n_data_epochs % self.publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    result.add_error(
                        f"`validate_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({self.publish_metrics_every_n_data_epochs}), but is {self.validate_every_n_data_epochs} instead."
                    )

        if self.checkpoint_every_n_steps is not None:
            if self.checkpoint_every_n_steps < 1:
                result.add_error(
                    "`checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_steps is not None:
                if self.checkpoint_every_n_steps % self.publish_metrics_every_n_steps != 0:  # fmt: skip
                    result.add_error(
                        f"`checkpoint_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({self.publish_metrics_every_n_steps}), but is {self.checkpoint_every_n_steps} instead."
                    )

        if self.checkpoint_every_n_data_epochs is not None:
            if self.checkpoint_every_n_data_epochs < 1:
                result.add_error(
                    "`checkpoint_every_n_data_epochs` must be greater than or equal to 1."
                )

            if self.publish_metrics_every_n_data_epochs is not None:
                if self.checkpoint_every_n_data_epochs % self.publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    result.add_error(
                        f"`checkpoint_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({self.publish_metrics_every_n_data_epochs}), but is {self.checkpoint_every_n_data_epochs} instead."
                    )

        if self.keep_last_n_checkpoints is not None:
            if self.keep_last_n_checkpoints < 1:
                result.add_error(
                    "`keep_last_n_checkpoints` must be greater than or equal to 1."
                )

        if self.keep_best_n_checkpoints is not None:
            if self.keep_best_n_checkpoints < 1:
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
            if self.keep_checkpoint_every_n_steps < 1:
                result.add_error(
                    "`keep_checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if self.checkpoint_every_n_steps is not None:
                if self.keep_checkpoint_every_n_steps % self.checkpoint_every_n_steps != 0:  # fmt: skip
                    result.add_error(
                        f"`keep_checkpoint_every_n_steps` must be a multiple of `checkpoint_every_n_steps` ({self.checkpoint_every_n_steps}), but is {self.keep_checkpoint_every_n_steps} instead."
                    )

        if self.publish_metrics_every_n_steps is not None:
            if self.publish_metrics_every_n_steps < 1:
                result.add_error(
                    "`publish_metrics_every_n_steps` must be greater than or equal to 1."
                )

        if self.publish_metrics_every_n_data_epochs is not None:
            if self.publish_metrics_every_n_data_epochs < 1:
                result.add_error(
                    "`publish_metrics_every_n_data_epochs` must be greater than or equal to 1."
                )

        return result


@dataclass(kw_only=True)
class EvaluatorSection:
    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    amp_dtype: DataType = torch.float32


@dataclass(kw_only=True)
class GeneratorSection:
    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    amp_dtype: DataType = torch.float32


@dataclass(kw_only=True)
class CommonSection:
    torch: TorchConfig = field(default_factory=lambda: TorchConfig())

    metric_recorders: MetricRecordersConfig = field(
        default_factory=lambda: MetricRecordersConfig()
    )

    profilers: ProfilersConfig = field(default_factory=lambda: ProfilersConfig())

    assets: AssetsConfig = field(default_factory=lambda: AssetsConfig())

    seed: int = 2

    debug: bool = False

    cluster: str = "auto"

    no_sweep_dir: bool = False

    sweep_format: str = "ws_{world_size}.{hash}"


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
class TorchConfig(Validatable):
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

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        budget = self.compiled_region_activation_memory_budget
        if budget < 0.0 or budget > 1.0:
            result.add_error(
                f"`compiled_region_activation_memory_budget` must be greater than or equal to 0.0 and less than or equal to 1.0, but is {budget} instead."
            )

        return result


@dataclass(kw_only=True)
class MetricRecordersConfig:
    tensorboard: TensorBoardConfig = field(default_factory=lambda: TensorBoardConfig())

    wandb: WandbConfig = field(default_factory=lambda: WandbConfig())


@dataclass(kw_only=True)
class TensorBoardConfig:
    enabled: bool = True


WandbResumeMode: TypeAlias = Literal["allow", "never", "must", "auto"]


@dataclass(kw_only=True)
class WandbConfig:
    """
    Holds the configuration of the Weights & Biases metric recorder.

    For more information, check out the official Weights & Biases
    `documentation <https://docs.wandb.ai/ref/python/sdk/functions/init/>`_.
    """

    enabled: bool = False
    """If ``True``, metrics will be written to Weights & Biases."""

    entity: str | None = None
    """The user or team that owns this job."""

    project: str | None = None
    """The name of the project under which this job will be logged."""

    run_id: str | None = "persistent"
    """
    The unique identifier of this job in the Weights & Biases dashboard.

    If set to ``persistent``, a random ID will be generated and saved in the
    output directory. Secondary invocations of the recipe will reuse the same ID.

    If ``None``, each recipe invocation will use a new random ID.
    """

    run_name: str | None = None
    """
    The display name of this job in Weights & Biases dashboard. If ``None``, a
    random two-word name will be generated.
    """

    group: str | None = None
    """
    The group name to organize individiual jobs as part of a larger experiment.
    """

    job_type: str | None = None
    """The type of the job such as "train", "eval"."""

    resume_mode: WandbResumeMode | None = None
    """
    The resume behavior when :attr`run_id` has already been used by another job.
    Refer to the Weights & Biases documentation for details.
    """


@dataclass(kw_only=True)
class ProfilersConfig:
    torch: TorchProfilerConfig = field(default_factory=lambda: TorchProfilerConfig())


@dataclass(kw_only=True)
class TorchProfilerConfig(Validatable):
    enabled: bool = False
    skip_n_steps: int = 4
    wait_n_steps: int = 0
    num_warmup_steps: int = 1
    num_active_steps: int = 4
    repeat: int = 1

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.skip_n_steps < 0:
            result.add_error("`skip_n_steps` must be greater than and equal to 0.")

        if self.wait_n_steps < 0:
            result.add_error("`wait_n_steps` must be greater than and equal to 0.")

        if self.num_warmup_steps < 0:
            result.add_error("`num_warmup_steps` must be greater than and equal to 0.")

        if self.num_active_steps < 1:
            result.add_error("`num_active_steps` must be greater than and equal to 1.")

        if self.repeat < 1:
            result.add_error("`repeat` must be greater than and equal to 1.")

        return result


@dataclass(kw_only=True)
class AssetsConfig:
    extra_paths: Sequence[Path] = field(default_factory=list)

    prev_checkpoint_dir: Path | None = None
    """If not ``None``, adds the specified path to the default asset store."""


ADAMW_OPTIMIZER: Final = "adamw"

ADAFACTOR_OPTIMIZER: Final = "adafactor"


@dataclass(kw_only=True)
class OptimizerSection(SupportsStructure):
    name: str = ADAMW_OPTIMIZER

    config: object = field(default_factory=lambda: AdamWConfig())

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, Optimizer, self.name, self.config
            )
        except ComponentNotKnownError:
            raise OptimizerNotKnownError(self.name) from None


@dataclass(kw_only=True)
class ParameterGroupConfig:
    """
    Represents the configuration of an optimizer parameter group.

    This configuration is meant to be subclassed and extended with additional
    fields for a particular optimizer. For an example, see :class:`AdamWGroupConfig`.
    """

    params: str | Sequence[str] = ".*"
    """
    The regular expression(s) used to select the parameters that belong to this
    group.
    """


@dataclass(kw_only=True)
class AdamWConfig:
    """Represents the configuration of :class:`AdamW`."""

    lr: float = 1e-3
    """The learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """
    The coefficients used for computing running averages of gradient and its
    square.
    """

    eps: float = 1e-8
    """The term added to the denominator to improve numerical stability."""

    weight_decay: float = 0.0
    """The weight decay coefficient."""

    amsgrad: bool = False
    """If ``True``, uses the AMSGrad variant."""

    maximize: bool = False
    """If ``True``, maximizes the parameters instead of minimizing."""

    capturable: bool = False
    """If ``True``, it is safe to capture this instance in a CUDA graph."""

    differentiable: bool = False
    """If ``True``, runs the optimizer step under autograd."""

    impl: Literal["auto", "foreach", "fused", "naive"] = "auto"
    """The implementation variant. See :class:`torch.optim.AdamW` for details."""

    groups: Sequence[AdamWGroupConfig] = field(default_factory=list)
    """
    The configuration of individual parameter groups. Each parameter will be
    assigned to the first group in the list that matches its name. Parameters
    that do not match any group will use the top-level configuration.
    """


@dataclass(kw_only=True)
class AdamWGroupConfig(ParameterGroupConfig):
    """Represents the configuration of an :class:`AdamW` parameter group."""

    lr: float | Default = default
    """The learning rate."""

    betas: tuple[float, float] | Default = default
    """
    The coefficients used for computing running averages of gradient and its
    square.
    """

    eps: float | Default = default
    """The term added to the denominator to improve numerical stability."""

    weight_decay: float | Default = default
    """The weight decay coefficient."""


@dataclass(kw_only=True)
class AdafactorConfig:
    """
    Represents the configuration of :class:`Adafactor`.

    Adafactor is an optimizer that saves memory compared to AdamW by using
    low-rank representation of gradient running averages. It is recommended to
    use higher learning rate with it.
    """

    lr: float = 1e-2
    """The learning rate."""

    beta2_decay: float = -0.8
    """
    The decay rate of beta2. beta2 refers to the coefficient used for computing
    the running average of the gradient squared.
    """

    eps: tuple[float | None, float] = (None, 0.001)
    """
    epsilon1 is the term added to the denominator of the update calculation to
    improve numerical stability. epsilon2 is the term used to avoid having too
    small a weight update when applying parameter scaling.
    """

    d: float = 1.0
    """The clipping threshold, used to avoid larger-than-desired updates."""

    weight_decay: float = 0.0
    """The weight decay coefficient."""

    foreach: bool | None = None
    """If ``True``, ``foreach`` implementation is used."""

    maximize: bool = False
    """If ``True``, maximizes the parameters instead of minimizing."""

    groups: Sequence[AdafactorGroupConfig] = field(default_factory=list)
    """
    The configuration of individual parameter groups. Each parameter will be
    assigned to the first group in the list that matches its name. Parameters
    that do not match any group will use the top-level configuration.
    """


@dataclass(kw_only=True)
class AdafactorGroupConfig(ParameterGroupConfig):
    """Represents the configuration of an :class:`Adafactor` parameter group."""

    lr: float | Default = default
    """The learning rate."""

    beta2_decay: float | Default = default
    """
    The decay rate of beta2. beta2 refers to the coefficient used for computing
    the running average of the gradient squared.
    """

    eps: tuple[float | None, float] | Default = default
    """
    epsilon1 is the term added to the denominator of the update calculation to
    improve numerical stability. epsilon2 is the term used to avoid having too
    small a weight update when applying parameter scaling.
    """

    d: float | Default = default
    """The clipping threshold, used to avoid larger-than-desired updates."""

    weight_decay: float | Default = default
    """The weight decay coefficient."""


PASSTHROUGH_LR: Final = "passthrough"

COSINE_ANNEALING_LR: Final = "cosine_annealing"

MYLE_LR: Final = "myle"

NOAM_LR: Final = "noam"

POLYNOMIAL_DECAY_LR: Final = "polynomial_decay"

TRI_STAGE_LR: Final = "tri_stage"


@dataclass(kw_only=True)
class LRSchedulerSection(SupportsStructure):
    name: str = COSINE_ANNEALING_LR

    config: object = field(default_factory=lambda: CosineAnnealingLRConfig())

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, LRScheduler, self.name, self.config
            )
        except ComponentNotKnownError:
            raise LRSchedulerNotKnownError(self.name) from None


@dataclass(kw_only=True)
class CosineAnnealingLRConfig(Validatable):
    """Represents the configuration of :class:`CosineAnnealingLR`."""

    cycle_len: int | None = None
    """
    The number of steps within the first cycle. If ``None``, will be set to
    ``num_steps - num_warmup_steps``.
    """

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    cycle_mul: float = 1.0
    """The factor to grow the length of each cycle."""

    lr_mul: float = 1.0
    """
    The factor to scale the base and final learning rate at the end of each cycle.
    """

    start_lr: float | Sequence[float] = 0.0
    """
    The initial warmup learning rate of all optimizer parameter groups or each
    group respectively.
    """

    final_lr: float | Sequence[float] | None = None
    """
    The final learning rate of all optimizer parameter groups or each group
    respectively. If ``None``, :attr:`final_lr_scale` will be used.
    """

    final_lr_scale: float | Sequence[float] | None = 0.2
    """
    The default learning rate of all optimizer parameter groups or each group
    respectively will be scaled by this value to determine the final learning
    rate. If ``None``, :attr:`final_lr` will be used.
    """

    def validate(self) -> ValidationResult:
        """:meta private:"""
        result = ValidationResult()

        if self.num_warmup_steps < 0:
            result.add_error("`num_warmup_steps` must be greater than or equal to 0.")

        if self.final_lr is not None:
            if self.final_lr_scale is not None:
                result.add_error(
                    "`final_lr` and `final_lr_scale` must not be specified at the same time."
                )
        elif self.final_lr_scale is None:
            result.add_error("`final_lr` or `final_lr_scale` must be specified.")

        return result


@dataclass(kw_only=True)
class MyleLRConfig(Validatable):
    """Represents the configuration of :class:`MyleLR`."""

    num_warmup_steps: int = 1
    """The number of warmup steps."""

    start_lr: float | Sequence[float] = 0.0
    """
    The initial warmup learning rate of all optimizer parameter groups or each
    group respectively.
    """

    def validate(self) -> ValidationResult:
        """:meta private:"""
        result = ValidationResult()

        if self.num_warmup_steps < 1:
            result.add_error("`num_warmup_steps` must be greater than or equal to 1.")

        return result


@dataclass(kw_only=True)
class NoamLRConfig(Validatable):
    """Represents the configuration of :class:`NoamLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    def validate(self) -> ValidationResult:
        """:meta private:"""
        result = ValidationResult()

        if self.num_warmup_steps < 0:
            result.add_error("`num_warmup_steps` must be greater than or equal to 0.")

        return result


@dataclass(kw_only=True)
class PolynomialDecayLRConfig(Validatable):
    """Represents the configuration of :class:`PolynomialDecayLR`."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    power: float = 1.0
    """The exponent of the polynomial used for decay."""

    start_lr: float | Sequence[float] = 0.0
    """
    The initial warmup learning rate of all optimizer parameter groups or each
    group respectively.
    """

    final_lr: float | Sequence[float] = 0.0
    """
    The final learning rate of all optimizer parameter groups or each group
    respectively.
    """

    def validate(self) -> ValidationResult:
        """:meta private:"""
        result = ValidationResult()

        if self.num_warmup_steps < 0:
            result.add_error("`num_warmup_steps` must be greater than or equal to 0.")

        return result


@dataclass(kw_only=True)
class TriStageLRConfig(Validatable):
    """Represents the configuration of :class:`TriStageLR`."""

    stage_ratio: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The ratios of warmup, hold, and decay stages, must add up to 1."""

    start_lr_scale: float | Sequence[float] = 0.01
    """
    The scale of the initial warm-up learning rate of all optimizer parameter
    groups or each group respectively.
    """

    final_lr_scale: float | Sequence[float] = 0.01
    """
    The scale of the final learning rate of all optimizer parameter groups or
    each group respectively.
    """

    def validate(self) -> ValidationResult:
        """:meta private:"""
        result = ValidationResult()

        ratio_sum = sum(self.stage_ratio)
        if not math.isclose(ratio_sum, 1.0):
            result.add_error(
                f"Sum of `stage_ratio` values must be 1.0, but is {ratio_sum} instead."
            )

        return result


SAMPLING_GENERATOR: Final = "sampling"

BEAM_SEARCH_GENERATOR: Final = "beam_search"


@dataclass(kw_only=True)
class SequenceGeneratorSection(SupportsStructure):
    name: str = SAMPLING_GENERATOR

    config: object = field(default_factory=lambda: SamplingConfig())

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, SequenceGenerator, self.name, self.config
            )
        except ComponentNotKnownError:
            raise SequenceGeneratorNotKnownError(self.name) from None


@dataclass(kw_only=True)
class Seq2SeqGeneratorSection(SupportsStructure):
    name: str = BEAM_SEARCH_GENERATOR

    config: object = field(default_factory=lambda: BeamSearchConfig())

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, Seq2SeqGenerator, self.name, self.config
            )
        except ComponentNotKnownError:
            raise SequenceGeneratorNotKnownError(self.name) from None


@dataclass(kw_only=True)
class SamplingConfig(Validatable):
    sampler: SamplerChoice = field(default_factory=lambda: SamplerChoice())

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[float, int] = 2048
    """The maximum generation length."""

    max_seq_len: int | None = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    compute_scores: bool = False
    """If ``True``, computes scores of generated sequences."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: int | None = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: int | None = 16
    """The sequence length capacity will be incremented by multiplies of this value."""

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.min_gen_len < 1:
            result.add_error("`min_gen_len` must be greater than or equal to 1.")

        if isinstance(self.max_gen_len, int):
            if self.max_gen_len < self.min_gen_len:
                result.add_error(
                    f"`max_gen_len` must be greater than or equal to `min_gen_len` ({self.min_gen_len}), but is {self.max_gen_len} instead."
                )

        if self.max_seq_len is not None:
            if self.max_seq_len < 1:
                result.add_error("`max_seq_len` must be greater than or equal to 1.")

        if self.temperature <= 0.0:
            result.add_error("`temperature` must be greater than 0.0.")

        if self.unk_penalty < 0.0:
            result.add_error("`unk_penalty` must be greater than or equal to 0.0.")

        if self.len_penalty < 1.0:
            result.add_error("`len_penalty` must be greater than or equal to 1.0.")

        if self.prefill_chunk_size is not None:
            if self.prefill_chunk_size < 1:
                result.add_error(
                    "`prefill_chunk_size` must be greater than or equal to 1."
                )

        if self.decode_capacity_increment is not None:
            if self.decode_capacity_increment < 1:
                result.add_error(
                    "`decode_capacity_increment` must be greater than or equal to 1."
                )

        return result


TOP_P_SAMPLER: Final = "top_p"

TOP_K_SAMPLER: Final = "top_k"


@dataclass(kw_only=True)
class SamplerChoice(SupportsStructure):
    name: str = TOP_P_SAMPLER

    config: object = field(default_factory=lambda: TopPSamplerConfig())

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, Sampler, self.name, self.config
            )
        except ComponentNotKnownError:
            raise SamplerNotKnownError(self.name) from None


@dataclass(kw_only=True)
class TopPSamplerConfig(Validatable):
    p: float = 0.9

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.p <= 0.0 or self.p > 1.0:
            result.add_error(
                f"`p` must be greater than 0.0 and less than or equal to 1.0, but is {self.p} instead."
            )

        return result


@dataclass(kw_only=True)
class TopKSamplerConfig(Validatable):
    k: int = 1

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.k < 1:
            result.add_error("`k` must be greater than or equal to 1.")

        return result


@dataclass(kw_only=True)
class BeamSearchConfig(Validatable):
    algo: BeamSearchAlgorithmChoice = field(
        default_factory=lambda: BeamSearchAlgorithmChoice()
    )
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[float, int] = 2048
    """The maximum generation length."""

    max_seq_len: int | None = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: int | None = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: int | None = 16
    """The sequence length capacity will be incremented by multiplies of this value."""

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.beam_size < 1:
            result.add_error("`beam_size` must be greater than or equal to 1.")

        if self.min_gen_len < 1:
            result.add_error("`min_gen_len` must be greater than or equal to 1.")

        if isinstance(self.max_gen_len, int):
            if self.max_gen_len < self.min_gen_len:
                result.add_error(
                    f"`max_gen_len` must be greater than or equal to `min_gen_len` ({self.min_gen_len}), but is {self.max_gen_len} instead."
                )

        if self.max_seq_len is not None:
            if self.max_seq_len < 1:
                result.add_error("`max_seq_len` must be greater than or equal to 1.")

        if self.temperature <= 0.0:
            result.add_error("`temperature` must be greater than 0.0.")

        if self.unk_penalty < 0.0:
            result.add_error("`unk_penalty` must be greater than or equal to 0.0.")

        if self.len_penalty < 1.0:
            result.add_error("`len_penalty` must be greater than or equal to 1.0.")

        if self.prefill_chunk_size is not None:
            if self.prefill_chunk_size < 1:
                result.add_error(
                    "`prefill_chunk_size` must be greater than or equal to 1."
                )

        if self.decode_capacity_increment is not None:
            if self.decode_capacity_increment < 1:
                result.add_error(
                    "`decode_capacity_increment` must be greater than or equal to 1."
                )

        return result


STANDARD_BEAM_SEARCH_ALGO: Final = "standard"


@dataclass(kw_only=True)
class BeamSearchAlgorithmChoice(SupportsStructure):
    name: str = STANDARD_BEAM_SEARCH_ALGO

    config: object | None = None

    @override
    def structure(self, resolver: DependencyResolver) -> None:
        try:
            self.config = structure_component_config(
                resolver, BeamSearchAlgorithm, self.name, self.config
            )
        except ComponentNotKnownError:
            raise BeamSearchAlgorithmNotKnownError(self.name) from None


def structure_component_config(
    resolver: DependencyResolver, kls: type[object], name: str, config: object
) -> object:
    component_manager = resolver.resolve(ComponentManager)

    try:
        return component_manager.structure_component_config(kls, name, config)
    except StructureError as ex:
        raise StructureError("`config` field cannot be structured.") from ex


class ConfigSectionNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Recipe configuration does not have a section named {name}.")

        self.name = name

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypeVar

import torch

from fairseq2.data_type import DataType
from fairseq2.dependency import DependencyResolver
from fairseq2.nn.data_parallel import FSDPGranularity
from fairseq2.utils.validation import ValidationError, ValidationResult


@dataclass(kw_only=True)
class ModelSection:
    name: str | None = None

    family: str | None = None

    arch: str | None = None

    config_overrides: object = None

    path: Path | None = None

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
class DatasetSectionBase:
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
class TokenizerSection:
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

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
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


ADAMW_OPTIMIZER: Final = "adamw"


@dataclass(kw_only=True)
class AdamWConfig:
    lr: float = 1e-3
    """The learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """The coefficients used for computing running averages of gradient and its
    square."""

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


@dataclass(kw_only=True)
class OptimizerSection:
    name: str = ADAMW_OPTIMIZER

    config: object = field(default_factory=AdamWConfig)


COSINE_ANNEALING_LR: Final = "cosine_annealing"


@dataclass(kw_only=True)
class CosineAnnealingLRConfig:
    cycle_len: int | None = None
    """The number of steps within the first cycle. If ``None``, will be set to
    ``num_steps - num_warmup_steps``."""

    num_warmup_steps: int = 0
    """The number of warmup steps."""

    cycle_mul: float = 1.0
    """The factor to grow the length of each cycle."""

    lr_mul: float = 1.0
    """The factor to scale the base and final learning rate at the end of each
    cycle."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    final_lr: float | None = None
    """The final learning rate. If ``None``, :attr:`final_lr_scale` will be used."""

    final_lr_scale: float | None = 0.2
    """
    The optimizer learning rate will be scaled by this value to determine the
    final learning rate. If ``None``, :attr:`final_lr` will be used.
    """

    def validate(self) -> None:
        result = ValidationResult()

        if self.final_lr is not None:
            if self.final_lr_scale is not None:
                result.add_error(
                    "`final_lr` and `final_lr_scale` must not be specified at the same time."
                )
        elif self.final_lr_scale is None:
            result.add_error("Either `final_lr` or `final_lr_scale` must be specified.")

        if result.has_error:
            raise ValidationError(
                "The cosine-annealing learning rate scheduler configuration has one or more validation errors:", result  # fmt: skip
            )


MYLE_LR: Final = "myle"


@dataclass(kw_only=True)
class MyleLRConfig:
    num_warmup_steps: int = 1
    """The number of warmup steps."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    def validate(self) -> None:
        result = ValidationResult()

        if self.num_warmup_steps == 0:
            result.add_error("`num_warmup_steps` must be greater than or equal to 1.")

        if result.has_error:
            raise ValidationError(
                "The Myle learning rate scheduler configuration has one or more validation errors:", result  # fmt: skip
            )


NOAM_LR: Final = "noam"


@dataclass(kw_only=True)
class NoamLRConfig:
    num_warmup_steps: int = 0
    """The number of warmup steps."""


POLYNOMIAL_DECAY_LR: Final = "polynomial_decay"


@dataclass(kw_only=True)
class PolynomialDecayLRConfig:
    num_warmup_steps: int = 0
    """The number of warmup steps."""

    power: float = 1.0
    """The exponent of the polynomial used for decay."""

    start_lr: float = 0.0
    """The initial warmup learning rate."""

    final_lr: float = 0.0
    """The final learning rate."""


TRI_STAGE_LR: Final = "tri_stage"


@dataclass(kw_only=True)
class TriStageLRConfig:
    stage_ratio: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The ratios of warmup, hold, and decay stages. Must add up to 1."""

    start_lr_scale: float = 0.01
    """The scale of the initial warm-up learning rate."""

    final_lr_scale: float = 0.01
    """The scale of the final learning rate."""

    def validate(self) -> None:
        result = ValidationResult()

        ratio_sum = sum(self.stage_ratio)
        if not math.isclose(ratio_sum, 1.0):
            result.add_error(
                f"The sum of `stage_ratio` values must be 1.0, but is {ratio_sum} instead."
            )

        if result.has_error:
            raise ValidationError(
                "The tri-stage learning rate scheduler configuratio has one or more validation errors:", result  # fmt: skip
            )


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

    score_metric: str | None = None

    checkpoint_after_n_steps: int = 0

    checkpoint_every_n_steps: int | None = None
    """The step interval at which to checkpoint."""

    checkpoint_after_n_data_epochs: int = 0

    checkpoint_every_n_data_epochs: int | None = None
    """The data epoch interval at which to checkpoint."""

    save_model_only: bool = False

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


JSONL_METRIC_RECORDER: Final = "jsonl"


@dataclass(kw_only=True)
class JsonlMetricRecorderConfig:
    enabled: bool = True


LOG_METRIC_RECORDER: Final = "log"


@dataclass(kw_only=True)
class LogMetricRecorderConfig:
    enabled: bool = True


TENSORBOARD_RECORDER: Final = "tensorboard"


@dataclass(kw_only=True)
class TensorBoardRecorderConfig:
    enabled: bool = True


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


TORCH_PROFILER: Final = "torch"


@dataclass(kw_only=True)
class TorchProfilerConfig:
    enabled: bool = False
    skip_n_steps: int = 4
    wait_n_steps: int = 0
    num_warmup_steps: int = 1
    num_active_steps: int = 4
    repeat: int = 1


SDPAVariant: TypeAlias = Literal[
    "torch",
    "torch_math",
    "torch_mem_efficient",
    "torch_flash",
    "flash2",
    "flash3",
    "flex",
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


# @dataclass(kw_only=True)
# class SequenceGeneratorSection:
#    name: str = SAMPLING_GENERATOR
#
#    config: object = field(default_factory=SamplingConfig)
#
#    batch_size: int = 1
#
#
# @dataclass(kw_only=True)
# class Seq2SeqGeneratorSection:
#    name: str = BEAM_SEARCH_GENERATOR
#
#    config: object = field(default_factory=BeamSearchConfig)
#
#    batch_size: int = 1


SAMPLING_GENERATOR: Final = "sampling"

SEQ2SEQ_SAMPLING_GENERATOR: Final = "seq2seq_sampling"


@dataclass(kw_only=True)
class SamplingConfig:
    sampler: ComponentSection = field(
        default_factory=lambda: ComponentSection(
            name=TOP_P_SAMPLER, config=TopPSamplerConfig()
        )
    )

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[int, int] = 2048
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


TOP_P_SAMPLER: Final = "top_p"


@dataclass(kw_only=True)
class TopPSamplerConfig:
    p: float = 0.9


TOP_K_SAMPLER: Final = "top_k"


@dataclass(kw_only=True)
class TopKSamplerConfig:
    k: int = 1


BEAM_SEARCH_GENERATOR: Final = "beam_search"

SEQ2SEQ_BEAM_SEARCH_GENERATOR: Final = "seq2seq_beam_search"


@dataclass(kw_only=True)
class BeamSearchConfig:
    algorithm: ComponentSection = field(
        default_factory=lambda: ComponentSection(
            name=STANDARD_BEAM_SEARCH_ALGO, config=None
        )
    )
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[int, int] = 2048
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


STANDARD_BEAM_SEARCH_ALGO = "standard_beam_search"


@dataclass
class ComponentSection:
    name: str
    config: object


def get_recipe_config(resolver: DependencyResolver) -> object:
    return resolver.resolve(object, key="recipe_config")


T = TypeVar("T")


def get_recipe_config_section(
    resolver: DependencyResolver, name: str, kls: type[T]
) -> T:
    config = get_recipe_config(resolver)

    section = getattr(config, name, None)
    if section is None:
        try:
            section = kls()
        except TypeError:
            raise LookupError(
                f"The recipe configuration does not have a section named '{name}'."
            ) from None
    elif not isinstance(section, kls):
        raise TypeError(
            f"The '{section}' recipe configuration section is expected to be of type `{kls}`, but is of type `{type(section)}` instead."
        )

    return section


def get_recipe_output_dir(resolver: DependencyResolver) -> Path:
    return resolver.resolve(Path, key="recipe_output_dir")

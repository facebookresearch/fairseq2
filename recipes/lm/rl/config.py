# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    COSINE_ANNEALING_LR,
    AdamWConfig,
    CommonSection,
    CompileOptions,
    CosineAnnealingLRConfig,
    DatasetSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TokenizerSection,
    TorchConfig,
    TrainerSection,
)

from .dataset import LM_RL_DATASET, DataReadOptions

@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"
    prompt_key: str = "prompt"
    tokenizer: str | None = None
    judgment_extractor: str | None = None


@dataclass(kw_only=True)
class RewardSection:
    name: str = "dummy"
    config: RewardModelConfig = field(default_factory=lambda: RewardModelConfig())
    vllm_reward_model_actor_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reward model."""

@dataclass(kw_only=True)
class GrpoUnitConfig:
    """Configuration for Generalized Reward-Paired Optimization (GRPO) finetuning.

    GRPO finetuning uses a policy model to generate diverse responses, which are then
    evaluated by a reward model. The policy is trained to maximize the expected reward
    while maintaining proximity to a reference model.
    """
    name: str = "grpo"

    loss_config: GrpoLossConfig = field(default_factory=lambda: GrpoLossConfig())
    """Configuration for GRPO loss computation, including rollout handling and regularization."""

    remote_policy_model_name: str = "vllm_policy"
    """Name of the Ray vLLM actor used to generate policy rollouts."""

    remote_reference_model_name: str | None = "vllm_reference"
    """Optional name of the Ray vLLM actor used as a reference model."""

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="dummy")
    )
    """Configuration for the reward function that evaluates generated rollouts."""

    sync_model_every_n_steps: int = 1
    """How often to sync the vLLM model with the policy that is trained. -1 disables syncing."""

    sync_ref_model_every_n_steps: int = -1
    """How often to sync the reference model with the policy. -1 disables syncing."""

@dataclass(kw_only=True)
class GrpoLossConfig:
    group_size: int = 4
    """Number of responses to sample per prompt for advantage computation.
    
    This value must match the 'n' parameter in the VLLM sampling params.
    """

    forward_group_size: int = 4
    """Maximum number of responses to process in a single forward pass.
    
    When group_size > forward_group_size, responses are processed in multiple micro-batches
    to reduce memory usage (similar to gradient accumulation). Each micro-batch processes
    forward_group_size responses and accumulates gradients until all group_size responses
    are processed.
    """

    beta: float = 0.001
    """The coefficient of regularization towards the reference model."""

    entropy_regularizer_scale: float = 0.0
    """Scale factor for entropy regularization term."""

    length_normalization: bool = True
    """If True, normalize loss by sequence length. If False, use sequence-level loss."""

    log_rollouts: bool = False
    """Log sample rollouts during training/validation."""

    validation_vllm_sampling_params: dict[str, object] = field(default_factory=lambda: {})
    """VLLM sampling params for validation. If empty, training params will be used."""


@dataclass(kw_only=True)
class VllmActorsSection:
    ray_cluster_ip_address: str | None = None
    ray_actors: List[VllmRayActorConfig | HFRayActorConfig] = field(default_factory=lambda: [VllmRayActorConfig()])


@dataclass(kw_only=True)
class RLDatasetSection(DatasetSection):
    family: str = "lm_rl"
    train_split: str = "default"
    valid_split: str | List[str] | None = None
    read_options: DataReadOptions = field(default_factory=lambda: DataReadOptions())
    

@dataclass(kw_only=True)
class RLTrainConfig:
    """
    RL recipe config
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )

    dataset: RLDatasetSection = field(
        default_factory=lambda: RLDatasetSection()
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            data_parallelism="fsdp",
        )
    )

    train_unit: GrpoUnitConfig = field(
        default_factory=lambda: GrpoUnitConfig()
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR, config=CosineAnnealingLRConfig(final_lr_scale=1.0)
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=5_000,
            checkpoint_every_n_steps=50,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=1,
        )
    )

    vllm: VllmActorsSection = field(default_factory=lambda: VllmActorsSection())

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class RayActorConfig:
    ray_actor_name: str = "dummy"
    backend: str = "vllm"  # vllm or hf
    num_replicas: int = 1
    init_update_process_group: bool = False
    blocking_initialization: bool = False


@dataclass(kw_only=True)
class VllmEngineArgs:
    model: str = "dummy"
    tokenizer: str = "dummy"
    task: str = "generate"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    model_impl: str = "auto"
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool = False
    hf_overrides: object = None
    dtype: str = "auto"

def get_default_sampling_params():
    from vllm import SamplingParams
    sp = SamplingParams()
    return {f: getattr(sp, f) for f in sp.__struct_fields__}

@dataclass(kw_only=True)
class VllmRayActorConfig(RayActorConfig):
    vllm_engine_args: VllmEngineArgs = field(default_factory=lambda: VllmEngineArgs())
    # vllm_sampling_params: Dict[str, object] = field(default_factory=lambda: {})
    vllm_sampling_params: Dict[str, object] = field(default_factory=lambda: get_default_sampling_params())
    sync_every_n_steps: int = 0
    """Sync model every n steps where step is fetched from the train unit. Default 0 means no synchronization. Values >0 require init_update_process_group to be set to True."""

@dataclass(kw_only=True)
class HFRayActorConfig(RayActorConfig):
    pipeline_name: str = ""
    tensor_parallel_size: int = 1
    blocking_initialization: bool = False
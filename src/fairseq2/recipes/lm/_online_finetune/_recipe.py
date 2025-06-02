# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import ray
import torch
import torch.distributed

from fairseq2.context import RuntimeContext
from fairseq2.datasets import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    InstructionDataset,
    InstructionPromptReadOptions,
)
from fairseq2.datasets.preference import (
    GENERIC_PREFERENCE_DATASET_FAMILY,
    PreferenceBatch,
)
from fairseq2.datasets.prompt import (
    GENERIC_PROMPT_DATASET_FAMILY,
    GenericPromptDataset,
    PromptDataset,
    PromptReadOptions,
)
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import COSINE_ANNEALING_LR, CosineAnnealingLRConfig
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_model,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    FsdpSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TrainerSection,
)
from fairseq2.recipes.lm._online_finetune._common import (
    OnlineCriterionSection,
    get_ray_actor,
)
from fairseq2.recipes.lm._online_finetune._grpo import GrpoFinetuneConfig
from fairseq2.recipes.lm._online_finetune._handler import (
    OnlineFinetuneUnitHandler,
    UnknownOnlineFinetuneUnitError,
)
from fairseq2.recipes.lm._online_finetune._online_dpo import (  # ONLINE_DPO_FINETUNE_UNIT,
    OnlineDpoFinetuneConfig,
)
from fairseq2.recipes.lm._online_finetune._grpo import (
    GrpoFinetuneConfig,
)

from fairseq2.recipes.lm._online_finetune._remote_vllm import (
    RemoteRayModelHandler,
    VllmRayActorConfig,
    HFRayActorConfig,
)
from fairseq2.recipes.trainer import Trainer
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class OnlineFinetuneConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )

    dataset: OnlineFinetuneDatasetSection = field(
        default_factory=lambda: OnlineFinetuneDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.bfloat16,
            data_parallelism="fsdp",
            fsdp=FsdpSection(fp32_reduce=True),
            activation_checkpointing=True,
        )
    )

    criterion: OnlineCriterionSection = field(
        default_factory=lambda: OnlineCriterionSection(name="grpo", config={})
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR, config=CosineAnnealingLRConfig(final_lr_scale=0.2)
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
class VllmActorsSection:
    ray_cluster_ip_address: str | None = None
    ray_actors: List[Union[VllmRayActorConfig, HFRayActorConfig]] | None = None


@dataclass(kw_only=True)
class OnlineFinetuneDatasetSection(DatasetSection):
    name: str = "foo"

    family: str = GENERIC_PROMPT_DATASET_FAMILY

    path: Path | None = (
        "/opt/hpcaas/.mounts/fs-08557fb804ac7e131/kulikov/llm_rl/data_wanswers_64.jsonl"
    )

    train_split: str = "default"

    valid_split: str | List[str] | None = None

    source_encode_mode: str = "prompt"
    """The encode mode for the prompt, determines what special tokens to add."""

    target_encode_mode: str = "prompt_response"
    """The encode mode for the target, determines what special tokens to add."""

    # mask_source_tokens: bool = True
    # """If ``False``, calculates loss on the `src` tokens as well as the `tgt` tokens."""

    min_seq_len: int = 1
    """The minimum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Shorter sequences will be dropped."""

    max_seq_len: int = 8192
    """The maximum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Longer sequences will be dropped."""

    # max_num_tokens: int = 8192 * 2
    # """The maximum number of total `src`, `tgt_chosen`, and `tgt_rejected` tokens per batch."""

    batch_size: int | None = 1
    """If not ``None``, ignores `max_num_tokens` and each batch will have `batch_size` examples."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1_000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""

    src_key: str = "src"


@dataclass(kw_only=True)
class DropoutConfig:
    dropout_p: float = 0.0


def register_online_finetune_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(OnlineFinetuneConfig)

    preset = registry.decorator

    @preset("llama3_1_instruct")
    def llama3_1_instruct() -> OnlineFinetuneConfig:
        config = OnlineFinetuneConfig()

        config.model.config = DropoutConfig()
        config.regime.validate_before_training = True

        return config

    @preset("llama3_1_instruct_grpo")
    def llama3_1_instruct() -> OnlineFinetuneConfig:
        config = OnlineFinetuneConfig()

        config.model.config = DropoutConfig()
        config.criterion.config = GrpoFinetuneConfig()

        return config

    @preset("llama3_1_instruct_constant_lr")
    def llama3_1_instruct_constant_lr() -> OnlineFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.optimizer.config, AdamWConfig)
        assert isinstance(config.lr_scheduler.config, CosineAnnealingLRConfig)

        config.lr_scheduler.config.final_lr = config.optimizer.config.lr

        return config


def load_online_finetuner(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer[PreferenceBatch]:
    config = structure(config, OnlineFinetuneConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_model(
        DecoderModel, context, config, output_dir, gangs, checkpoint_manager
    )

    # TODO(balioglu): investigate!
    # The memory efficient SDPA implementation in PyTorch is not stable when
    # used with padded inputs.
    enable_memory_efficient_torch_sdpa(model.module, False)

    optimizer = create_optimizer(context, config, model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    dataset = load_dataset(PromptDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # initialize ray and vllm actors
    ray.init(
        address=f"ray://{config.vllm.ray_cluster_ip_address}:10001",
        namespace="vllm_workers",
    )

    vllm_actors = {}
    # go over actor configs and initialize all of them
    for actor_config in config.vllm.ray_actors:
        log.info(f"Setting up '{actor_config.ray_actor_name}' vllm actor")
        actor = RemoteRayModelHandler().create(gangs=gangs, actor_config=actor_config)
        vllm_actors[actor_config.ray_actor_name] = actor

    # Initialize the train unit.
    unit_handlers = context.get_registry(OnlineFinetuneUnitHandler)

    try:
        unit_handler = unit_handlers.get(config.criterion.name)
    except LookupError:
        raise UnknownOnlineFinetuneUnitError(config.criterion.name) from None

    unit = unit_handler.create(model, gangs, config, vllm_actors)
    print(f"rank {gangs.root.rank} here")

    try:
        if unit._sync_vllm_model_every_n_steps >= 0:
            unit.maybe_sync_models(force_sync_vllm=True)
    except AttributeError:
        raise RuntimeError("Train unit does not support maybe_sync_models")

    valid_unit = unit_handler.create(model, gangs, config, vllm_actors)

    batching: Batching

    if config.dataset.batch_size is not None:
        batching = StaticBatching(config.dataset.batch_size)
    else:
        raise ValueError

    # estimate batch repeat for microbatching
    repeat_batch_n_times = 1
    gradient_accumulation = config.trainer.gradient_accumulation
    if unit.display_name == "GRPO":
        if unit._loss_config.group_size > unit._loss_config.forward_group_size:
            repeat_batch_n_times = int(
                unit._loss_config.group_size / unit._loss_config.forward_group_size
            )
            # adjust grad accum so that it assumed repeated batches
            gradient_accumulation = (
                repeat_batch_n_times * config.trainer.gradient_accumulation
            )
        elif unit._loss_config.group_size < unit._loss_config.forward_group_size:
            raise RuntimeError(
                " GRPO forward_group_size must be smaller than group_size"
            )

    read_options = PromptReadOptions(
        batching=batching,
        example_shuffle_window=config.dataset.example_shuffle_window,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        source_encode_mode=config.dataset.source_encode_mode,
        seed=seed,
        extras=config.dataset.extras,
        src_key=config.dataset.src_key,
        repeat_batch_n_times=repeat_batch_n_times,
    )

    data_reader = dataset.create_reader(
        config.dataset.train_split,
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    if config.dataset.valid_split:
        valid_batching = StaticBatching(32)
        valid_read_options = PromptReadOptions(
            batching=valid_batching,
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=1,
            num_prefetch=config.dataset.num_prefetch,
            source_encode_mode=config.dataset.source_encode_mode,
            max_num_batches=500,  ## TODO make confifurable ?
            seed=seed,
            extras=config.dataset.extras,
            src_key=config.dataset.src_key,
        )
        if isinstance(config.dataset.valid_split, list):
            valid_data_readers = []
            for valid_split in config.dataset.valid_split:
                vdr = dataset.create_reader(
                    valid_split,
                    tokenizer,
                    gangs.dp,
                    config.dataset.min_seq_len,
                    config.dataset.max_seq_len,
                    valid_read_options,
                )
                valid_data_readers.append(vdr)
        elif isinstance(config.dataset.valid_split, str):
            valid_data_readers = [
                dataset.create_reader(
                    config.dataset.valid_split,
                    tokenizer,
                    gangs.dp,
                    config.dataset.min_seq_len,
                    config.dataset.max_seq_len,
                    valid_read_options,
                )
            ]
        else:
            raise ValueError(
                "valid split has to be either list of splits or single split"
            )
    else:
        valid_data_readers = []

    if len(valid_data_readers) > 0:
        valid_units = [valid_unit] * len(valid_data_readers)
    else:
        valid_units = []

    seed += 1

    return create_trainer(
        context,
        config,
        output_dir,
        unit,
        data_reader,
        valid_units,
        valid_data_readers,
        gangs,
        checkpoint_manager,
        optimizer,
        lr_scheduler,
        seed,
    )

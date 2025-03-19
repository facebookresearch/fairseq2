# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed

from fairseq2.context import RuntimeContext
from fairseq2.datasets import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.preference import (
    GENERIC_PREFERENCE_DATASET_FAMILY,
    PreferenceBatch,
    PreferenceDataset,
    PreferenceReadOptions,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import COSINE_ANNEALING_LR, CosineAnnealingLRConfig
from fairseq2.recipes import Trainer
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
from fairseq2.recipes.lm._preference_finetune._common import POCriterionSection
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DPO_FINETUNE_UNIT,
    DpoFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._handler import (
    POFinetuneUnitHandler,
    UnknownPOFinetuneUnitError,
)
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class POFinetuneConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )

    dataset: POFinetuneDatasetSection = field(
        default_factory=lambda: POFinetuneDatasetSection()
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

    criterion: POCriterionSection = field(
        default_factory=lambda: POCriterionSection(
            name=DPO_FINETUNE_UNIT, config=DpoFinetuneConfig()
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1, impl="fused"
            ),
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
            checkpoint_every_n_steps=1_000,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=10,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class POFinetuneDatasetSection(DatasetSection):
    name: str = "gsm8k_dpo"

    family: str = GENERIC_PREFERENCE_DATASET_FAMILY

    path: Path | None = None

    source_encode_mode: str = "prompt"
    """The encode mode for the prompt, determines what special tokens to add."""

    target_encode_mode: str = "prompt_response"
    """The encode mode for the target, determines what special tokens to add."""

    mask_source_tokens: bool = True
    """If ``False``, calculates loss on the `src` tokens as well as the `tgt` tokens."""

    min_seq_len: int = 1
    """The minimum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Shorter sequences will be dropped."""

    max_seq_len: int = 8192
    """The maximum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Longer sequences will be dropped."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of total `src`, `tgt_chosen`, and `tgt_rejected` tokens per batch."""

    batch_size: int | None = None
    """If not ``None``, ignores `max_num_tokens` and each batch will have `batch_size` examples."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1_000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_po_finetune_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(POFinetuneConfig)

    preset = registry.decorator

    @preset("llama3_1_instruct")
    def llama3_1_instruct() -> POFinetuneConfig:
        return POFinetuneConfig()

    @preset("llama3_1_instruct_constant_lr")
    def llama3_1_instruct_constant_lr() -> POFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.optimizer.config, AdamWConfig)
        assert isinstance(config.lr_scheduler.config, CosineAnnealingLRConfig)

        config.lr_scheduler.config.final_lr = config.optimizer.config.lr

        return config

    @preset("llama3_1_instruct_lr_anneal_0")
    def llama3_1_instruct_lr_anneal_0() -> POFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.lr_scheduler.config, CosineAnnealingLRConfig)

        config.lr_scheduler.config.final_lr = 0.0

        return config

    @preset("llama3_1_70b_instruct")
    def llama3_1_70b_instruct() -> POFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.criterion.config, DpoFinetuneConfig)

        config.model.name = "llama3_1_70b_instruct"
        config.gang.tensor_parallel_size = 8
        config.criterion.config.reference_model.name = "llama3_1_70b_instruct"

        return config


def load_po_finetuner(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer[PreferenceBatch]:
    config = structure(config, POFinetuneConfig)

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

    dataset = load_dataset(PreferenceDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # Initialize the train unit.
    unit_handlers = context.get_registry(POFinetuneUnitHandler)

    try:
        unit_handler = unit_handlers.get(config.criterion.name)
    except LookupError:
        raise UnknownPOFinetuneUnitError(config.criterion.name) from None

    unit = unit_handler.create(model, gangs, config)

    batching: Batching

    if config.dataset.batch_size is not None:
        batching = StaticBatching(config.dataset.batch_size)
    else:
        batching = LengthBatching(config.dataset.max_num_tokens)

    read_options = PreferenceReadOptions(
        batching=batching,
        example_shuffle_window=config.dataset.example_shuffle_window,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        mask_source_tokens=config.dataset.mask_source_tokens,
        source_encode_mode=config.dataset.source_encode_mode,
        target_encode_mode=config.dataset.target_encode_mode,
        seed=seed,
        extras=config.dataset.extras,
    )

    data_reader = dataset.create_reader(
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_trainer(
        context,
        config,
        output_dir,
        unit,
        data_reader,
        [],
        [],
        gangs,
        checkpoint_manager,
        optimizer,
        lr_scheduler,
        seed,
    )

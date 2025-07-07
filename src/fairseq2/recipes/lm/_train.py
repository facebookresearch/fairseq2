# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SequenceBatch
from fairseq2.datasets.jsonl import JSONL_DATASET_FAMILY
from fairseq2.datasets.text import TextDataset, TextReadOptions
from fairseq2.device import CPU
from fairseq2.metrics import MetricBag
from fairseq2.models.clm import CausalLM
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import COSINE_ANNEALING_LR, CosineAnnealingLRConfig
from fairseq2.recipes import Model, Trainer, TrainUnit
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_model,
    setup_torch,
    setup_training_gangs,
)
from fairseq2.recipes.config import (
    CommonSection,
    CompileOptionsSection,
    DatasetSection,
    FSDPSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TextTokenizerSection,
    TorchSection,
    TrainerSection,
)
from fairseq2.recipes.metrics import update_nll_loss, update_seq_batch_metrics
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class CausalLMTrainConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="llama",
            arch="llama3_8b",
            compile=True,
            compile_options=CompileOptionsSection(fullgraph=True, dynamic=False),
        )
    )

    dataset: TextDatasetSection = field(default_factory=lambda: TextDatasetSection())

    tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="llama3")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.bfloat16,
            data_parallelism="fsdp",
            fsdp=FSDPSection(version="v2", fp32_reduce=True),
            max_grad_norm=1.0,
            gc_every_n_steps=1000,
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, impl="fused"
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR,
            config=CosineAnnealingLRConfig(
                num_warmup_steps=2000, start_lr=1e-30, final_lr_scale=0.01
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=48_000,
            validate_every_n_steps=4000,
            checkpoint_every_n_steps=1000,
            keep_last_n_checkpoints=3,
            keep_checkpoint_every_n_steps=4000,
            publish_metrics_every_n_steps=10,
        )
    )

    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchSection(
                default_sdpa="flash2", compiled_region_activation_memory_budget=0.9
            )
        )
    )


@dataclass(kw_only=True)
class TextDatasetSection(DatasetSection):
    name: str | None = None

    family: str = JSONL_DATASET_FAMILY  # PARQUET_TEXT_DATASET_FAMILY

    path: Path | None = None

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    num_prefetch: int = 10
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_clm_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(CausalLMTrainConfig)

    preset = registry.decorator

    @preset("llama3_8b")
    def llama3_8b() -> CausalLMTrainConfig:
        return CausalLMTrainConfig()


def load_clm_trainer(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer:
    config = structure(config, CausalLMTrainConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_training_gangs(context, config.gang, config.trainer)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_model(
        CausalLM,
        context,
        config.model,
        config.trainer,
        output_dir,
        gangs,
        checkpoint_manager,
    )

    optimizer = create_optimizer(context, config.optimizer, model)

    lr_scheduler = create_lr_scheduler(
        context, config.lr_scheduler, config.regime, optimizer
    )

    dataset = load_dataset(TextDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.tokenizer)

    # Initialize the unit.
    unit = CausalLMTrainUnit(model)

    batching = LengthBatching(config.dataset.max_num_tokens)

    read_options = TextReadOptions(
        batching=batching,
        num_accumulate=config.trainer.grad_accumulation.num_batches,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
    )

    text_encoder = tokenizer.create_encoder(mode="default")

    min_seq_len = 1

    data_reader = dataset.create_reader(
        text_encoder,
        tokenizer.vocab_info.pad_idx,
        gangs.dp,
        min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_trainer(
        context,
        config.trainer,
        config.regime,
        config.common,
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
        hyper_params=config,
    )


@final
class CausalLMTrainUnit(TrainUnit[SequenceBatch]):
    _model: Model

    def __init__(self, model: Model) -> None:
        self._model = model

    @override
    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, None]:
        batch, target_batch = batch.as_auto_regressive()

        seqs, seqs_layout = batch.as_input()

        nll_loss = self._model.module(
            seqs, seqs_layout, targets=target_batch.seqs, reduction="mean"
        )

        update_nll_loss(metric_bag, nll_loss)

        update_seq_batch_metrics(metric_bag, batch)

        return nll_loss, None

    @property
    @override
    def model(self) -> Model:
        return self._model

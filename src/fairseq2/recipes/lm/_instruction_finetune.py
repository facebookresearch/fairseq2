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
from torch.nn import Module
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import Batching, LengthBatching, StaticBatching, SyncMode
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    InstructionDataset,
    InstructionReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import COSINE_ANNEALING_LR, CosineAnnealingLRConfig
from fairseq2.recipes.common import (
    check_model_type,
    compile_model,
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    load_model,
    load_text_tokenizer,
    prepare_model,
    register_extra_asset_paths,
    save_checkpoint_card,
    setup_gangs,
    wrap_data_parallel,
)
from fairseq2.recipes.config import (
    DatasetSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TrainerSection,
    TrainRecipeConfig,
)
from fairseq2.recipes.evaluator import AbstractEvalUnit
from fairseq2.recipes.metrics import SequenceMetricBag
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class InstructionFinetuneConfig(TrainRecipeConfig):
    """Holds the configuration of a language model instruction-finetuning task."""

    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )

    dataset: InstructionFinetuneDatasetSection = field(
        default_factory=lambda: InstructionFinetuneDatasetSection()
    )

    tokenizer: str | None = None

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.bfloat16, data_parallelism="fsdp", activation_checkpointing=True
        )
    )

    # Optimizer, Learning Rate
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
            validate_every_n_steps=100,
            checkpoint_every_n_steps=1_000,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=10,
        )
    )


@dataclass(kw_only=True)
class InstructionFinetuneDatasetSection(DatasetSection):
    name: str = "foo"  # TODO: change!

    family: str = GENERIC_INSTRUCTION_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "default"

    valid_split: str | None = None

    source_encode_mode: str = "prompt"
    """The encode mode for the prompt, determines what special tokens to add."""

    target_encode_mode: str = "prompt_response"
    """The encode mode for the target, determines what special tokens to add."""

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    batch_size: int | None = None
    """
    If not ``None``, ignores ``max_num_tokens`` and each batch will have
    ``batch_size`` examples.
    """

    max_num_valid_tokens: int | None = None
    """The maximum number of tokens per validation batch."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


@dataclass(kw_only=True)
class DropoutConfig:
    dropout_p: float = 0.0


def register_instruction_finetune_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(InstructionFinetuneConfig)

    preset = registry.decorator

    @preset("llama3_1_instruct")
    def llama3_1_instruct() -> InstructionFinetuneConfig:
        config = InstructionFinetuneConfig()

        config.model.config = DropoutConfig()

        return config

    @preset("llama3_1_instruct_constant_lr")
    def llama3_1_instruct_constant_lr() -> InstructionFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.optimizer.config, AdamWConfig)
        assert isinstance(config.lr_scheduler.config, CosineAnnealingLRConfig)

        config.lr_scheduler.config.final_lr = config.optimizer.config.lr

        return config

    @preset("llama3_1_instruct_lr_anneal_0")
    def llama3_1_instruct_lr_anneal_0() -> InstructionFinetuneConfig:
        config = llama3_1_instruct()

        assert isinstance(config.lr_scheduler.config, CosineAnnealingLRConfig)

        config.lr_scheduler.config.final_lr = 0.0

        return config

    @preset("llama3_1_70b_instruct")
    def llama3_1_70b_instruct() -> InstructionFinetuneConfig:
        config = llama3_1_instruct()

        config.model.name = "llama3_1_70b_instruct"
        config.gang.tensor_parallel_size = 8

        return config

    @preset("llama2_7b_chat")
    def llama2_7b_chat() -> InstructionFinetuneConfig:
        config = llama3_1_instruct()

        config.model.name = "llama2_7b_chat"
        config.dataset.max_seq_len = 4096
        config.dataset.max_num_tokens = 4096 * 2
        config.dataset.max_num_valid_tokens = 4096 * 2

        return config

    @preset("llama2_70b_chat")
    def llama2_70b_chat() -> InstructionFinetuneConfig:
        config = llama2_7b_chat()

        config.model.name = "llama2_70b_chat"
        config.gang.tensor_parallel_size = 8

        return config


def load_instruction_finetuner(
    context: RuntimeContext, config: InstructionFinetuneConfig, output_dir: Path
) -> Trainer[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    dataset = load_dataset(InstructionDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model.name, config.tokenizer)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    log_model_config(log, config.model.config)

    model = load_model(DecoderModel, context, config, gangs, checkpoint_manager)

    dp_model = wrap_data_parallel(context, config, model, gangs, checkpoint_manager)

    dp_model = prepare_model(context, config, dp_model, gangs)

    log_model(log, dp_model, gangs)

    if config.trainer.torch_compile:
        dp_model = compile_model(context, config.model, dp_model)

    # TODO(balioglu): investigate!
    # The memory efficient SDPA implementation in PyTorch is not stable when
    # used with padded inputs.
    enable_memory_efficient_torch_sdpa(dp_model, False)

    save_checkpoint_card(context, config, checkpoint_manager, config.tokenizer)

    optimizer = create_optimizer(context, config, dp_model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    # Initialize the unit.
    criterion = InstructionFinetuneCriterion(dp_model)

    unit = InstructionFinetuneUnit(criterion, gangs)

    batching: Batching

    if config.dataset.batch_size is not None:
        batching = StaticBatching(config.dataset.batch_size)
    else:
        batching = LengthBatching(config.dataset.max_num_tokens)

    read_options = InstructionReadOptions(
        batching=batching,
        example_shuffle_window=config.dataset.example_shuffle_window,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        source_encode_mode=config.dataset.source_encode_mode,
        target_encode_mode=config.dataset.target_encode_mode,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.train_split,
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    # Initialize the validation unit.
    if config.dataset.valid_split is not None:
        valid_unit = InstructionLossEvalUnit(criterion, gangs)

        max_num_tokens = (
            config.dataset.max_num_valid_tokens or config.dataset.max_num_tokens
        )

        batching = LengthBatching(max_num_tokens)

        read_options = InstructionReadOptions(
            batching=batching,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            source_encode_mode=config.dataset.source_encode_mode,
            target_encode_mode=config.dataset.target_encode_mode,
            seed=seed,
        )

        valid_data_reader = dataset.create_reader(
            config.dataset.valid_split,
            tokenizer,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        valid_units = [valid_unit]

        valid_data_readers = [valid_data_reader]
    else:
        valid_units = []

        valid_data_readers = []

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


@final
class InstructionFinetuneUnit(AbstractTrainUnit[SequenceBatch]):
    _criterion: InstructionFinetuneCriterion
    _metric_bag: SequenceMetricBag

    def __init__(self, criterion: InstructionFinetuneCriterion, gangs: Gangs) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = SequenceMetricBag(gangs.dp)

    @override
    def __call__(self, batch: SequenceBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> SequenceMetricBag:
        return self._metric_bag


@final
class InstructionLossEvalUnit(AbstractEvalUnit[SequenceBatch]):
    _criterion: InstructionFinetuneCriterion
    _metric_bag: SequenceMetricBag

    def __init__(self, criterion: InstructionFinetuneCriterion, gangs: Gangs) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = SequenceMetricBag(gangs.dp)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> SequenceMetricBag:
        return self._metric_bag


@final
class InstructionFinetuneCriterion:
    _model: Module

    def __init__(self, model: Module) -> None:
        check_model_type(model, DecoderModel)

        self._model = model

    def __call__(
        self, batch: SequenceBatch, metric_bag: SequenceMetricBag
    ) -> tuple[Tensor, int]:
        input_batch, target_batch = as_auto_regressive_input(batch)

        output = self._forward(input_batch)

        loss = output.compute_loss(
            target_batch.seqs, loss_mask=target_batch.target_mask
        )

        metric_bag.update_nll_loss(target_batch, loss)

        metric_bag.update_batch_metrics(target_batch)

        return loss, target_batch.num_target_elements()

    def _forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    def model(self) -> Module:
        return self._model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.context import RuntimeContext
from fairseq2.datasets import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.preference import PreferenceDataset, PreferenceReadOptions
from fairseq2.device import CPU
from fairseq2.models.clm import CausalLM
from fairseq2.optim import AdamWConfig
from fairseq2.optim.lr_scheduler import CosineAnnealingLRConfig
from fairseq2.recipes import Trainer
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
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.recipes.lm._preference_finetune._config import (
    POCriterionSection,
    POFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._dpo import (
    DPO_FINETUNE_UNIT,
    DpoFinetuneConfig,
)
from fairseq2.recipes.lm._preference_finetune._handler import (
    POFinetuneUnitHandler,
    UnknownPOFinetuneUnitError,
)


def register_po_finetune_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(POFinetuneConfig)

    preset = registry.decorator

    @preset("llama3_1_instruct")
    def llama3_1_instruct() -> POFinetuneConfig:
        return POFinetuneConfig(
            criterion=POCriterionSection(
                name=DPO_FINETUNE_UNIT, config=DpoFinetuneConfig()
            )
        )

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

        if config.criterion.config.reference_model is not None:
            config.criterion.config.reference_model.name = "llama3_1_70b_instruct"

        return config


def load_po_finetuner(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer:
    config = structure(config, POFinetuneConfig)

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

    dataset = load_dataset(PreferenceDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.tokenizer)

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
        num_accumulate=config.trainer.grad_accumulation.num_batches,
        num_prefetch=config.dataset.num_prefetch,
        mask_source_tokens=config.dataset.mask_source_tokens,
        source_encode_mode=config.dataset.source_encode_mode,
        target_encode_mode=config.dataset.target_encode_mode,
        seed=seed,
        extras=config.dataset.extras,
        chat_mode=config.dataset.chat_mode,
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

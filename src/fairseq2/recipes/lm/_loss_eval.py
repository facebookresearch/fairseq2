# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    InstructionDataset,
    InstructionReadOptions,
)
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.common import (
    broadcast_model,
    compile_eval_model,
    create_evaluator,
    load_dataset,
    load_eval_model,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
)
from fairseq2.recipes.config import DatasetSection, EvalRecipeConfig, EvaluatorSection
from fairseq2.recipes.evaluator import Evaluator
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneCriterion,
    InstructionLossEvalUnit,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class LMLossEvalConfig(EvalRecipeConfig):
    """Holds configuration of the perplexity evaluator recipe"""

    model: str = "llama3_1_8b"

    dataset: LMLossEvalDatasetSection = field(
        default_factory=lambda: LMLossEvalDatasetSection()
    )

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.bfloat16)
    )


@dataclass(kw_only=True)
class LMLossEvalDatasetSection(DatasetSection):
    name: str = "foo"

    family: str = GENERIC_INSTRUCTION_DATASET_FAMILY

    path: Path | None = None

    split: str = "default"

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_elements: int = 8192 * 2
    """The maximum number of elements per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


def register_lm_loss_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(LMLossEvalConfig)

    preset = registry.decorator

    @preset("llama3_1_base_eval")
    def llama3_1_base_eval() -> LMLossEvalConfig:
        return LMLossEvalConfig()


@torch.inference_mode()
def load_lm_loss_evaluator(
    context: RuntimeContext, config: LMLossEvalConfig, output_dir: Path
) -> Evaluator[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    dataset = load_dataset(InstructionDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    model = load_eval_model(
        DecoderModel,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        mixed_precision=config.evaluator.amp,
    )

    broadcast_model(config.model, model, gangs)

    remove_parametrizations(model)

    log_model(log, model, gangs)

    if config.evaluator.torch_compile:
        model = compile_eval_model(context, config.model, model)

    # Initialize the unit.
    criterion = InstructionFinetuneCriterion(model)

    unit = InstructionLossEvalUnit(criterion, gangs)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = InstructionReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_evaluator(
        context, config, output_dir, [unit], [data_reader], gangs, seed
    )

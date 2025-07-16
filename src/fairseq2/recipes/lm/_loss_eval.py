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
from fairseq2.device import CPU
from fairseq2.models.clm import CausalLM
from fairseq2.recipes import Evaluator
from fairseq2.recipes.common import (
    create_evaluator,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
    setup_torch,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
    TextTokenizerSection,
)
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneCriterion,
    InstructionLossEvalUnit,
)


@dataclass(kw_only=True)
class CausalLMLossEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="llama3_1_8b")
    )

    dataset: CausalLMLossEvalDatasetSection = field(
        default_factory=lambda: CausalLMLossEvalDatasetSection()
    )

    tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.bfloat16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class CausalLMLossEvalDatasetSection(DatasetSection):
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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_clm_loss_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(CausalLMLossEvalConfig)

    preset = registry.decorator

    @preset("llama3_1_base_eval")
    def llama3_1_base_eval() -> CausalLMLossEvalConfig:
        return CausalLMLossEvalConfig()


@torch.inference_mode()
def load_clm_loss_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator:
    config = structure(config, CausalLMLossEvalConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_gangs(context, config.gang)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        CausalLM,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
    )

    dataset = load_dataset(InstructionDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.tokenizer)

    # Initialize the unit.
    criterion = InstructionFinetuneCriterion(model.module)

    unit = InstructionLossEvalUnit(model, criterion)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = InstructionReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    units = [unit]

    data_readers = [data_reader]

    seed += 1

    return create_evaluator(
        context,
        config.evaluator,
        config.common,
        output_dir,
        units,
        data_readers,
        gangs,
        seed,
        hyper_params=config,
    )

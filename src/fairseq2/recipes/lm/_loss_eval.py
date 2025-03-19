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
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes import Evaluator
from fairseq2.recipes.common import (
    create_evaluator,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
)
from fairseq2.recipes.lm._instruction_finetune import (
    InstructionFinetuneCriterion,
    InstructionLossEvalUnit,
)
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class LMLossEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="llama3_1_8b")
    )

    dataset: LMLossEvalDatasetSection = field(
        default_factory=lambda: LMLossEvalDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.bfloat16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_lm_loss_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(LMLossEvalConfig)

    preset = registry.decorator

    @preset("llama3_1_base_eval")
    def llama3_1_base_eval() -> LMLossEvalConfig:
        return LMLossEvalConfig()


@torch.inference_mode()
def load_lm_loss_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator[SequenceBatch]:
    config = structure(config, LMLossEvalConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        DecoderModel,
        context,
        config.model.name,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
        config.evaluator.torch_compile,
    )

    dataset = load_dataset(InstructionDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # Initialize the unit.
    criterion = InstructionFinetuneCriterion(model)

    unit = InstructionLossEvalUnit(criterion, gangs)

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

    seed += 1

    return create_evaluator(
        context, config, output_dir, [unit], [data_reader], gangs, seed
    )

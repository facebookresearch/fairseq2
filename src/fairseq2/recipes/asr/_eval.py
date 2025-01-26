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
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, AsrDataset, AsrReadOptions
from fairseq2.error import SetupError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.models.asr import AsrModel
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.asr._common import AsrCriterion, AsrMetricBag, AsrScorer
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
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.log import log_model
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.file import FileMode
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class AsrEvalConfig(EvalRecipeConfig):
    """Holds the configuration of an ASR model evaluation task."""

    model: str = "wav2vec2_asr_base_10h"

    dataset: AsrEvalDatasetSection = field(
        default_factory=lambda: AsrEvalDatasetSection()
    )

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )


@dataclass(kw_only=True)
class AsrEvalDatasetSection(DatasetSection):
    name: str | None = "librilight_asr_10h"

    family: str = GENERIC_ASR_DATASET_FAMILY

    path: Path | None = None

    split: str = "test_other"

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


def register_asr_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(AsrEvalConfig)

    preset = registry.decorator

    @preset("wav2vec2_base_10h")
    def base_10h() -> AsrEvalConfig:
        return AsrEvalConfig()


@torch.inference_mode()
def load_asr_evaluator(
    context: RuntimeContext, config: AsrEvalConfig, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    dataset = load_dataset(AsrDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    model = load_eval_model(
        AsrModel,
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
    if gangs.tp.rank == 0:
        file_system = context.file_system

        rank = gangs.dp.rank

        ref_file = output_dir.joinpath(f"transcriptions/rank_{rank}.ref.txt")
        hyp_file = output_dir.joinpath(f"transcriptions/rank_{rank}.hyp.txt")

        try:
            file_system.make_directory(ref_file.parent)
        except OSError as ex:
            raise SetupError(
                f"The '{ref_file.parent}' output directory cannot be created. See the nested exception for details."
            ) from ex

        try:
            ref_fp = file_system.open_text(ref_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{ref_file}' output file cannot be created. See the nested exception for details."
            ) from ex

        try:
            hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{hyp_file}' output file cannot be created. See the nested exception for details."
            ) from ex
    else:
        ref_fp = None
        hyp_fp = None

    scorer = AsrScorer(tokenizer, ref_output_stream=ref_fp, hyp_output_stream=hyp_fp)

    criterion = AsrCriterion(model, scorer)

    unit = AsrEvalUnit(criterion, gangs)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = AsrReadOptions(
        batching=batching,
        dtype=config.evaluator.dtype,
        normalize_audio=config.dataset.normalize_audio,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        tokenizer,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    seed += 1

    return create_evaluator(
        context, config, output_dir, [unit], [data_reader], gangs, seed
    )


@final
class AsrEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    _criterion: AsrCriterion
    _metric_bag: AsrMetricBag

    def __init__(self, criterion: AsrCriterion, gangs: Gangs) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = AsrMetricBag(gangs.dp, train=False)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> AsrMetricBag:
        return self._metric_bag

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import torch
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, Seq2SeqBatch, SyncMode
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, AsrDataset, AsrReadOptions
from fairseq2.device import CPU
from fairseq2.file_system import FileMode
from fairseq2.metrics import MetricBag
from fairseq2.models.asr import AsrModel
from fairseq2.recipes import Evaluator, EvalUnit, Model, RecipeError, UnitError
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

from fairseq2.recipes.asr._metrics import update_asr_batch_metrics, update_ctc_loss
from fairseq2.recipes.asr._scorer import AsrScorer


@dataclass(kw_only=True)
class AsrEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_asr_base_10h")
    )

    dataset: AsrEvalDatasetSection = field(
        default_factory=lambda: AsrEvalDatasetSection()
    )

    tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="librispeech_asr")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_asr_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(AsrEvalConfig)

    preset = registry.decorator

    @preset("wav2vec2")
    def wav2vec2() -> AsrEvalConfig:
        return AsrEvalConfig()


@torch.inference_mode()
def load_asr_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator:
    config = structure(config, AsrEvalConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_gangs(context, config.gang)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        AsrModel,
        context,
        config.model,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
    )

    dataset = load_dataset(AsrDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.tokenizer)

    # Initialize the unit.
    if gangs.tp.rank == 0:
        file_system = context.file_system

        rank = gangs.dp.rank

        try:
            ref_file = output_dir.joinpath(f"transcriptions/rank_{rank}.ref.txt")
            hyp_file = output_dir.joinpath(f"transcriptions/rank_{rank}.hyp.txt")

            try:
                file_system.make_directory(ref_file.parent)
            except OSError as ex:
                raise UnitError(
                    f"The '{ref_file.parent}' output directory cannot be created. See the nested exception for details."
                ) from ex

            try:
                ref_fp = file_system.open_text(ref_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise UnitError(
                    f"The '{ref_file}' output file cannot be created. See the nested exception for details."
                ) from ex

            try:
                hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise UnitError(
                    f"The '{hyp_file}' output file cannot be created. See the nested exception for details."
                ) from ex
        except UnitError as ex:
            raise RecipeError(
                "The evaluator unit cannot be initialized. See the nested exception for details."
            ) from ex
    else:
        ref_fp = None
        hyp_fp = None

    scorer = AsrScorer(tokenizer, ref_output_stream=ref_fp, hyp_output_stream=hyp_fp)

    unit = AsrEvalUnit(model, scorer)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = AsrReadOptions(
        batching=batching,
        dtype=config.evaluator.dtype,
        normalize_audio=config.dataset.normalize_audio,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
    )

    data_reader = dataset.create_reader(
        config.dataset.split,
        tokenizer,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
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


@final
class AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    _model: Model
    _scorer: AsrScorer

    def __init__(self, model: Model, scorer: AsrScorer) -> None:
        self._model = model

        self._scorer = scorer

    @override
    def __call__(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        source_seqs, source_seqs_layout = batch.as_source_input()
        target_seqs, target_seqs_layout = batch.as_target_input()

        ctc_loss, logits, logits_layout = self._model.module(
            source_seqs,
            source_seqs_layout,
            target_seqs,
            target_seqs_layout,
            return_logits=True,
        )

        update_ctc_loss(metric_bag, ctc_loss, batch.batch_size)

        update_asr_batch_metrics(metric_bag, batch)

        self._scorer(batch, logits, logits_layout, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        self._scorer.process_metric_values(values)

    @property
    @override
    def model(self) -> Model:
        return self._model

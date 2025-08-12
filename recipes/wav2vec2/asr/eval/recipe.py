# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import final

from typing_extensions import override

from fairseq2.datasets import Seq2SeqBatch, register_dataset_family
from fairseq2.error import raise_operational_system_error
from fairseq2.evaluator import Evaluator, EvalUnit
from fairseq2.file_system import FileMode
from fairseq2.metrics import MetricBag
from fairseq2.model import Model
from fairseq2.recipe.base import EvalRecipe, RecipeContext
from fairseq2.runtime.dependency import DependencyContainer

from ..data import (
    WAV2VEC2_ASR_DATASET,
    Wav2Vec2AsrDataset,
    Wav2Vec2AsrDatasetConfig,
    open_wav2vec2_asr_dataset,
)
from .default_config import Wav2Vec2AsrEvalConfig
from .scorer import Wav2Vec2AsrScorer


@final
class Wav2Vec2AsrEvalRecipe(EvalRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            WAV2VEC2_ASR_DATASET,
            Wav2Vec2AsrDataset,
            Wav2Vec2AsrDatasetConfig,
            opener=open_wav2vec2_asr_dataset,
        )

    @override
    def create_evaluator(self, context: RecipeContext) -> Evaluator:
        config = context.config_as(Wav2Vec2AsrEvalConfig)

        # Evaluation equivalent of the training criterion
        scorer = self._init_asr_scorer(context)

        dataset = context.dataset_as(Wav2Vec2AsrDataset)

        if config.dataset.valid_split == None:
            raise ValueError(
                "Wav2Vec2AsrDatasetConfig.valid_split must be defined for evaluation but is `None`."
            )

        # Initialize validation units
        eval_units = []
        eval_data_readers = []
        eval_splits = config.dataset.valid_split.split(",")

        for split in eval_splits:
            eval_unit = Wav2Vec2AsrEvalUnit(scorer)
            eval_units.append(eval_unit)

            eval_data_reader = dataset.create_reader(
                split,
                context.tokenizer,
                context.gangs,
                min_audio_len=config.dataset.min_audio_len,
                max_audio_len=config.dataset.max_audio_len,
                # Batching parameters
                batching_strategy=config.dataset.batching_strategy,
                batch_size=config.dataset.batch_size,
                num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
                max_num_elements=config.dataset.max_num_elements,
                # Audio processing parameters
                dtype=config.evaluator.dtype,
                normalize_audio=config.dataset.normalize_audio,
                no_padding=config.dataset.no_padding,
                npc=config.dataset.npc,
                # Shuffling and performance parameters
                example_shuffle_window=config.dataset.example_shuffle_window,
                batch_shuffle_window=config.dataset.batch_shuffle_window,
                num_accumulate=config.dataset.num_accumulate,
                num_prefetch=config.dataset.num_prefetch,
                drop_remainder=config.dataset.drop_remainder,
                sync_batches=config.dataset.sync_batches,
                sync_mode=config.dataset.sync_mode,
                seed=context.next_seed(),
                max_num_batches=config.dataset.max_num_batches,
                cached_fd_count=config.dataset.cached_fd_count,
            )
            eval_data_readers.append(eval_data_reader)

        return context.create_evaluator(eval_units, eval_data_readers)

    def _init_asr_scorer(self, context: RecipeContext) -> Wav2Vec2AsrScorer:
        """Glues the context with the pure evaluation scorer implementation. Only TP=0 and every DP rank score."""

        if context.gangs.tp.rank == 0:
            file_system = context.file_system

            rank = context.gangs.dp.rank

            ref_file = context.output_dir.joinpath(
                f"transcriptions/rank_{rank}.ref.txt"
            )
            hyp_file = context.output_dir.joinpath(
                f"transcriptions/rank_{rank}.hyp.txt"
            )

            try:
                file_system.make_directory(ref_file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)
            try:
                ref_fp = file_system.open_text(ref_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)
            try:
                hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)
        else:
            ref_fp = None
            hyp_fp = None

        return Wav2Vec2AsrScorer(
            tokenizer=context.tokenizer,
            model=context.model,
            ref_output_stream=ref_fp,
            hyp_output_stream=hyp_fp,
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrEvalConfig


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """
    wav2vec2 ASR evaluation unit. Replaces the evaluation runner from the training recipe with dedicated WER calcualtion.

    Note the process_metric_values override which allows us to modify the logged values.
    """

    _scorer: Wav2Vec2AsrScorer

    def __init__(self, scorer: Wav2Vec2AsrScorer) -> None:
        self._scorer = scorer

    @override
    def __call__(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._scorer(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        """Accessor function to modify the logged values"""
        return self._scorer.process_metric_values(values)

    @property
    @override
    def model(self) -> Model:
        return self._scorer.model

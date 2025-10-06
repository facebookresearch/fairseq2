# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast, final

from torch import Tensor
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import Seq2SeqBatch, SyncMode
from fairseq2.gang import GangError, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.recipe import (
    EvalUnit,
    RecipeContext,
    RecipeModel,
    Trainer,
    TrainRecipe,
    TrainUnit,
)
from fairseq2.recipe.error import RecipeError
from fairseq2.runtime.dependency import DependencyContainer

from .criterion import Wav2Vec2AsrCriterion
from .data import (
    WAV2VEC2_ASR_DATASET,
    Wav2Vec2AsrDataset,
    Wav2Vec2AsrDatasetConfig,
    open_wav2vec2_asr_dataset,
)
from .default_config import Wav2Vec2AsrRecipeConfig
from .wer_calculator import WerCalculator


@final
class Wav2Vec2AsrRecipe(TrainRecipe):
    """wav2vec2 ASR training recipe."""

    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            WAV2VEC2_ASR_DATASET,
            Wav2Vec2AsrDataset,
            Wav2Vec2AsrDatasetConfig,
            opener=open_wav2vec2_asr_dataset,
        )

    def prepare_ctc_training_from_encoder(
        self, context: RecipeContext, asr_module: Wav2Vec2AsrModel
    ) -> None:
        log.info("Initializing the asr model with pretrained ssl model (encoder only).")
        w2v2_model = context.bootstrap_model("pretrained_model")
        w2v2_module = cast(Wav2Vec2Model, w2v2_model.module)

        share_parameters(w2v2_module.encoder_frontend, asr_module.encoder_frontend)
        share_parameters(w2v2_module.encoder, asr_module.encoder)
        if asr_module.masker is not None:
            share_parameters(w2v2_module.masker, asr_module.masker)

        del w2v2_model

    @override
    def prepare_model(self, context: RecipeContext, model: RecipeModel) -> RecipeModel:
        """Initialize ASR model based on configuration:
        - model.name defined: Load pretrained model for finetuning
        - model.arch + pretrained_model.name defined: Train CTC from pretrained encoder
        """
        asr_module: Wav2Vec2AsrModel = cast(Wav2Vec2AsrModel, model.module)

        if model.newly_initialized:
            self.prepare_ctc_training_from_encoder(
                context=context, asr_module=asr_module
            )

        # Make sure that the final projection layer is instantiated along with
        # the pretrained parameters if it was on the meta device.
        if context.gangs.dp.rank == 0:
            to_device(asr_module, context.gangs.root.device)

        try:
            context.gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        # Always freeze feature extractor.
        freeze_parameters(asr_module.encoder_frontend.feature_extractor)

        return model

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        """
        When starting training from a pretrained model, the scaffolding happens in `self.prepare_model`.
        """
        config = context.config.as_(Wav2Vec2AsrRecipeConfig)

        criterion = Wav2Vec2AsrCriterion(context.model)

        unit = Wav2Vec2AsrTrainUnit(
            criterion, config.trainer.freeze_encoder_for_n_steps
        )
        dataset = context.default_dataset.as_(Wav2Vec2AsrDataset)

        if config.dataset.train_split is None:
            raise RecipeError(
                "`dataset.train_split` must be defined for training but is `None`."
            )

        seed = config.common.seed

        data_reader = dataset.create_reader(
            config.dataset.train_split,
            context.default_tokenizer,
            context.gangs,
            min_audio_len=config.dataset.min_audio_len,
            max_audio_len=config.dataset.max_audio_len,
            # Batching parameters
            batching_strategy=config.dataset.batching_strategy,
            batch_size=config.dataset.batch_size,
            max_num_elements=config.dataset.max_num_elements,
            num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
            # Audio processing parameters
            dtype=config.dataset.dtype,
            normalize_audio=config.dataset.normalize_audio,
            no_padding=config.dataset.no_padding,
            npc=config.dataset.npc,
            # Shuffling and performance parameters
            example_shuffle_window=config.dataset.example_shuffle_window,
            batch_shuffle_window=config.dataset.batch_shuffle_window,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            num_prefetch=config.dataset.num_prefetch,
            drop_remainder=config.dataset.drop_remainder,
            sync_batches=config.dataset.sync_batches,
            sync_mode=config.dataset.sync_mode,
            seed=seed,
            max_num_batches=config.dataset.max_num_batches,
            cached_fd_count=config.dataset.cached_fd_count,
        )

        valid_units = []
        valid_data_readers = []

        if config.dataset.valid_split is not None:
            # Support multiple validation splits
            valid_splits = config.dataset.valid_split.split(",")
            valid_criterion = Wav2Vec2AsrCriterion(
                model=context.model, wer_calculator=WerCalculator.from_context(context)
            )
            for split in valid_splits:
                seed += 1

                valid_unit = Wav2Vec2AsrEvalUnit(valid_criterion)
                valid_units.append(valid_unit)

                # Same parameters as training but with validation-specific settings
                valid_data_reader = dataset.create_reader(
                    split,
                    context.default_tokenizer,
                    context.gangs,
                    min_audio_len=config.dataset.min_audio_len,
                    max_audio_len=config.dataset.max_audio_len,
                    # Batching parameters
                    batching_strategy=config.dataset.batching_strategy,
                    batch_size=config.dataset.batch_size,
                    max_num_elements=config.dataset.max_num_elements,
                    num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
                    # Audio processing parameters
                    dtype=config.dataset.dtype,
                    normalize_audio=config.dataset.normalize_audio,
                    no_padding=config.dataset.no_padding,
                    npc=config.dataset.npc,
                    # Shuffling and performance parameters
                    example_shuffle_window=1,  # No pre-batch shuffling
                    batch_shuffle_window=1,  # No batch shuffling
                    num_accumulate=1,  # No accumulation
                    num_prefetch=config.dataset.num_prefetch,
                    drop_remainder=config.dataset.drop_remainder,
                    sync_batches=config.dataset.sync_batches,
                    sync_mode=SyncMode.UNTIL_LAST,  # Wait for all processes
                    seed=seed,
                    max_num_batches=config.dataset.max_num_batches,
                    cached_fd_count=config.dataset.cached_fd_count,
                )
                valid_data_readers.append(valid_data_reader)

        return context.create_trainer(
            unit, data_reader, valid_units, valid_data_readers
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrRecipeConfig


@final
class Wav2Vec2AsrTrainUnit(TrainUnit[Seq2SeqBatch]):
    """ASR training unit with encoder freezing logic."""

    _criterion: Wav2Vec2AsrCriterion
    _freeze_encoder_for_n_steps: int
    _frozen: bool

    def __init__(
        self, criterion: Wav2Vec2AsrCriterion, freeze_encoder_for_n_steps: int
    ) -> None:
        """
        :param criterion: The ASR criterion holding the model.
        :param freeze_encoder_for_n_steps: The encoder will be frozen for this number of steps.
        """
        self._criterion = criterion
        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps
        self._frozen = False

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def set_step_nr(self, step_nr: int) -> None:
        """Gradually unfreeze encoder during training for stability.
        Freezes encoder/masker for first N steps, then unfreezes while keeping feature extractor frozen.
        """
        base_module = cast(Wav2Vec2AsrModel, self._criterion.model.base_module)

        if step_nr <= self._freeze_encoder_for_n_steps:
            if self._frozen:
                return

            if step_nr == 1:
                log.info(f"Freezing the encoder for the first {self._freeze_encoder_for_n_steps} steps.")  # fmt: skip

            # Freeze encoder components
            freeze_parameters(base_module.encoder_frontend)
            freeze_parameters(base_module.encoder)

            if base_module.masker is not None:
                freeze_parameters(base_module.masker)

            self._frozen = True
        else:
            if not self._frozen:
                return

            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info(f"Unfreezing the encoder after step {step_nr - 1}.")

            # Unfreeze all parameters
            freeze_parameters(base_module, False)

            # Always keep feature extractor frozen
            freeze_parameters(base_module.encoder_frontend.feature_extractor)

            self._frozen = False

    @override
    def process_batch(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """ASR evaluation unit for validation during training."""

    _criterion: Wav2Vec2AsrCriterion

    def __init__(self, criterion: Wav2Vec2AsrCriterion) -> None:
        self._criterion = criterion

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

        metric_bag.add("wer", WerMetric())

    @override
    def process_batch(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        return self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model

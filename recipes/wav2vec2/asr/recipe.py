# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.datasets import Seq2SeqBatch, SyncMode, register_dataset_family
from fairseq2.error import OperationalError
from fairseq2.evaluator import EvalUnit
from fairseq2.gang import GangError
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.model import Model
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.eval_model import load_reference_model
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.trainer import Trainer, TrainUnit
from recipes.wav2vec2.asr.wer_calculator import WerCalculator

from .criterion import Wav2Vec2AsrCriterion
from .data import (
    WAV2VEC2_ASR_DATASET,
    Wav2Vec2AsrDataset,
    Wav2Vec2AsrDatasetConfig,
    open_wav2vec2_asr_dataset,
)
from .default_config import Wav2Vec2AsrConfig


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

    @staticmethod
    def fix_wav2vec2_position_encoder_weights(module: Module) -> Module:
        """Fix legacy checkpoints: convert deprecated weight_norm (g/v) to single weight.
        Required for compatibility when loading old wav2vec2 checkpoints into ASR model.

        Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html
        """
        conv = module.encoder_frontend.pos_encoder.conv  # type: ignore
        module.encoder_frontend.pos_encoder.conv = torch.nn.utils.remove_weight_norm(  # type: ignore
            conv  # type: ignore
        )
        return module

    @override
    def prepare_model(self, context: RecipeContext, model: Model) -> Model:
        """Initialize ASR model with pretrained wav2vec2 encoder weights.
        Shares encoder_frontend, encoder, and masker parameters from pretrained model to enable
        transfer learning while keeping the final projection layer trainable.
        """
        asr_module = cast(Wav2Vec2AsrModel, model.module)
        asr_module = Wav2Vec2AsrRecipe.fix_wav2vec2_position_encoder_weights(asr_module)

        if model.newly_initialized:
            log.info("Initializing the asr model with the pretrained wav2vec2 model.")

            # Loading the `pretrained_model` section (== config.pretrained_model)
            # TODO: cirquit - fix this after rebasing on the new version
            w2v2_model = load_reference_model(context.resolver, "pretrained_model")
            w2v2_module = cast(Wav2Vec2Model, w2v2_model.module)

            share_parameters(w2v2_module.encoder_frontend, asr_module.encoder_frontend)  # type: ignore
            share_parameters(w2v2_module.encoder, asr_module.encoder)  # type: ignore
            if asr_module.masker is not None:
                share_parameters(w2v2_module.masker, asr_module.masker)  # type: ignore

            del w2v2_model  # ~50% memory savings right here

            # Make sure that the final projection layer is instantiated along with
            # the pretrained parameters if it was on the meta device.
            if context.gangs.dp.rank == 0:
                to_device(asr_module, context.gangs.root.device)

            try:
                context.gangs.root.barrier()
            except GangError as ex:
                raise OperationalError(
                    "The collective barrier after the pretrained model load operation has failed. See the nested exception for details."
                ) from ex

        # Always freeze feature extractor
        freeze_parameters(asr_module.encoder_frontend.feature_extractor)  # type: ignore

        return model

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        """
        The pretrained model loading happens in `self.prepare_model`.
        """
        config = context.config_as(Wav2Vec2AsrConfig)

        criterion = Wav2Vec2AsrCriterion(context.model)

        unit = Wav2Vec2AsrTrainUnit(
            criterion, config.trainer.freeze_encoder_for_n_steps
        )
        dataset = context.dataset_as(Wav2Vec2AsrDataset)

        if config.dataset.train_split is None:
            raise ValueError(
                "Wav2Vec2AsrDatasetConfig.train_split must be defined for training but is `None`."
            )

        data_reader = dataset.create_reader(
            config.dataset.train_split,
            context.tokenizer,
            context.gangs,
            min_audio_len=config.dataset.min_audio_len,
            max_audio_len=config.dataset.max_audio_len,
            # Batching parameters
            batching_strategy=config.dataset.batching_strategy,
            batch_size=config.dataset.batch_size,
            max_num_elements=config.dataset.max_num_elements,
            num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
            # Audio processing parameters
            dtype=config.trainer.dtype,
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

        valid_units = []
        valid_data_readers = []

        if config.dataset.valid_split is not None:
            # Support multiple validation splits
            valid_splits = config.dataset.valid_split.split(",")
            valid_criterion = Wav2Vec2AsrCriterion(
                model=context.model, wer_calculator=WerCalculator.from_context(context)
            )
            for split in valid_splits:
                valid_unit = Wav2Vec2AsrEvalUnit(valid_criterion)
                valid_units.append(valid_unit)

                # Same parameters as training but with validation-specific settings
                valid_data_reader = dataset.create_reader(
                    split,
                    context.tokenizer,
                    context.gangs,
                    min_audio_len=config.dataset.min_audio_len,
                    max_audio_len=config.dataset.max_audio_len,
                    # Batching parameters
                    batching_strategy=config.dataset.batching_strategy,
                    batch_size=config.dataset.batch_size,
                    max_num_elements=config.dataset.max_num_elements,
                    num_seqs_multiple_of=config.dataset.num_seqs_multiple_of,
                    # Audio processing parameters
                    dtype=config.trainer.dtype,
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
                    seed=context.next_seed(),
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
        return Wav2Vec2AsrConfig


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
    def set_step_nr(self, step_nr: int) -> None:
        """Gradually unfreeze encoder during training for stability.
        Freezes encoder/masker for first N steps, then unfreezes while keeping feature extractor frozen.
        """
        base_module = cast(Wav2Vec2AsrModel, self._criterion.model.base_module)

        if step_nr <= self._freeze_encoder_for_n_steps:
            if self._frozen:
                return

            if step_nr == 1:
                log.info(
                    f"Freezing the encoder for the first {self._freeze_encoder_for_n_steps} steps."
                )

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
    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """ASR evaluation unit for validation during training."""

    _criterion: Wav2Vec2AsrCriterion

    def __init__(self, criterion: Wav2Vec2AsrCriterion) -> None:
        self._criterion = criterion

    @override
    def __call__(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        return self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

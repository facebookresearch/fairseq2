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
from torch import Tensor
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, AsrDataset, AsrReadOptions
from fairseq2.gang import Gang
from fairseq2.logging import log
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import TRI_STAGE_LR, TriStageLRConfig
from fairseq2.recipes.common import (
    compile_model,
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    load_eval_model,
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
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.recipes.wav2vec2.asr._common import (
    Wav2Vec2AsrCriterion,
    Wav2Vec2AsrMetricBag,
    Wav2Vec2AsrScorer,
)
from fairseq2.recipes.wav2vec2.asr._eval import Wav2Vec2AsrEvalUnit
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainConfig(TrainRecipeConfig):
    """Holds the configuration of a wav2vec 2.0 ASR model training task."""

    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2_asr", arch="base_10h")
    )

    pretrained_model: str = "wav2vec2_base"
    """The name or path to the asset card of the wav2vec 2.0 model to finetune."""

    dataset: Wav2Vec2AsrTrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainDatasetSection()
    )

    tokenizer: str | None = "librispeech_asr"

    trainer: Wav2Vec2AsrTrainerSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainerSection(
            dtype=torch.float16, gradient_accumulation=4
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER, config=AdamWConfig(lr=5e-05, betas=(0.9, 0.98))
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=TRI_STAGE_LR,
            config=TriStageLRConfig(
                stage_ratio=(0.1, 0.4, 0.5), start_lr_scale=0.01, final_lr_scale=0.05
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=20_000,
            score_metric="wer",
            lower_score_better=True,
            validate_after_n_steps=10_000,
            validate_every_n_steps=1_000,
            publish_metrics_every_n_steps=200,
        )
    )


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainDatasetSection(DatasetSection):
    name: str | None = "librilight_asr_10h"

    family: str = GENERIC_ASR_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"

    valid_split: str | None = "dev_other"

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    example_shuffle_window: int = 0
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainerSection(TrainerSection):
    freeze_encoder_for_n_steps: int = 10_000
    """The encoder will be frozen for this number of steps."""


def register_wav2vec2_asr_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Wav2Vec2AsrTrainConfig)

    preset = registry.decorator

    @preset("base_10h")
    def base_10h() -> Wav2Vec2AsrTrainConfig:
        return Wav2Vec2AsrTrainConfig()

    @preset("base_100h")
    def base_100h() -> Wav2Vec2AsrTrainConfig:
        config = base_10h()

        assert isinstance(config.optimizer.config, AdamWConfig)

        config.model.arch = "base_100h"
        config.dataset.name = "librispeech_asr_100h"
        config.optimizer.config.lr = 0.00003
        config.trainer.freeze_encoder_for_n_steps = 0
        config.regime.num_steps = 50_000

        return config

    @preset("large_10h")
    def large_10h() -> Wav2Vec2AsrTrainConfig:
        config = base_10h()

        assert isinstance(config.optimizer.config, AdamWConfig)

        config.model.arch = "large_10h"
        config.pretrained_model = "wav2vec2_large"
        config.dataset.max_audio_len = 640_000
        config.dataset.max_num_elements = 1_280_000
        config.trainer.gradient_accumulation = 5
        config.optimizer.config.lr = 0.0001

        return config

    @preset("large_100h")
    def large_100h() -> Wav2Vec2AsrTrainConfig:
        config = large_10h()

        assert isinstance(config.optimizer.config, AdamWConfig)

        config.model.arch = "large_100h"
        config.dataset.name = "librispeech_asr_100h"
        config.optimizer.config.lr = 0.00003
        config.regime.num_steps = 50_000

        return config


def load_wav2vec2_asr_trainer(
    context: RuntimeContext, config: Wav2Vec2AsrTrainConfig, output_dir: Path
) -> Trainer[Seq2SeqBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    dataset = load_dataset(AsrDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model.name, config.tokenizer)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    log_model_config(log, config.model.config)

    model = load_model(Wav2Vec2AsrModel, context, config, gangs, checkpoint_manager)

    # If we start the training with an empty ASR model, use the weights of a
    # pretrained wav2vec 2.0 model.
    if config.model.name is None and not checkpoint_manager.has_checkpoint():
        pt_model = load_eval_model(
            Wav2Vec2Model,
            context,
            config.pretrained_model,
            gangs,
            config.trainer.dtype,
            mixed_precision=config.trainer.mixed_precision is not None,
        )

        share_parameters(pt_model.encoder_frontend, model.encoder_frontend)
        share_parameters(pt_model.encoder, model.encoder)

        if model.masker is not None:
            share_parameters(pt_model.masker, model.masker)

        del pt_model

        # Make sure that the final projection layer is instantiated along with
        # the pretrained parameters if it was on the meta device.
        if gangs.dp.rank == 0:
            to_device(model, gangs.root.device)

        gangs.root.barrier()

    # We never train the feature extractor.
    freeze_parameters(model.encoder_frontend.feature_extractor)

    static_graph = config.trainer.freeze_encoder_for_n_steps == 0

    dp_model = wrap_data_parallel(
        context, config, model, gangs, checkpoint_manager, static_graph
    )

    dp_model = prepare_model(context, config, dp_model, gangs)

    log_model(log, dp_model, gangs)

    if config.trainer.torch_compile:
        model = compile_model(context, config.model, model)

    save_checkpoint_card(context, config, checkpoint_manager, config.tokenizer)

    optimizer = create_optimizer(context, config, dp_model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    # Initialize the train unit.
    criterion = Wav2Vec2AsrCriterion(dp_model)

    unit = Wav2Vec2AsrTrainUnit(
        criterion, gangs.dp, config.trainer.freeze_encoder_for_n_steps
    )

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = AsrReadOptions(
        batching=batching,
        dtype=config.trainer.dtype,
        normalize_audio=config.dataset.normalize_audio,
        example_shuffle_window=config.dataset.example_shuffle_window,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.train_split,
        tokenizer,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    seed += 1

    # Initialize the validation unit.
    if config.dataset.valid_split is not None:
        valid_scorer = Wav2Vec2AsrScorer(tokenizer)

        valid_criterion = Wav2Vec2AsrCriterion(dp_model, valid_scorer)

        valid_unit = Wav2Vec2AsrEvalUnit(valid_criterion, gangs)

        read_options = AsrReadOptions(
            batching=batching,
            dtype=config.trainer.dtype,
            normalize_audio=config.dataset.normalize_audio,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
        )

        valid_data_reader = dataset.create_reader(
            config.dataset.valid_split,
            tokenizer,
            gangs.dp,
            config.dataset.min_audio_len,
            config.dataset.max_audio_len,
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
class Wav2Vec2AsrTrainUnit(AbstractTrainUnit[Seq2SeqBatch]):
    _criterion: Wav2Vec2AsrCriterion
    _freeze_encoder_for_n_steps: int
    _metric_bag: Wav2Vec2AsrMetricBag

    def __init__(
        self,
        criterion: Wav2Vec2AsrCriterion,
        gang: Gang,
        freeze_encoder_for_n_steps: int,
    ) -> None:
        """
        :param freeze_encoder_for_n_steps: The encoder will be frozen for this
            number of steps.
        """
        super().__init__(criterion.model)

        self._criterion = criterion

        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps

        self._metric_bag = Wav2Vec2AsrMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @override
    def set_step_nr(self, step_nr: int) -> None:
        if isinstance(self._model, Wav2Vec2AsrModel):
            model = self._model
        else:
            model = self._model.module  # DDP or FSDP

        if step_nr <= self._freeze_encoder_for_n_steps:
            if step_nr == 1:
                log.info("Freezing the encoder for the first {} steps.", self._freeze_encoder_for_n_steps)  # fmt: skip

            freeze_parameters(model.encoder_frontend)
            freeze_parameters(model.encoder)

            if model.masker is not None:
                freeze_parameters(model.masker)
        else:
            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info("Unfreezing the encoder after step {}.", step_nr - 1)

            freeze_parameters(model, False)

            # We never train the feature extractor.
            freeze_parameters(model.encoder_frontend.feature_extractor)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2AsrMetricBag:
        return self._metric_bag

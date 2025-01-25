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
from fairseq2.datasets.speech import (
    GENERIC_SPEECH_DATASET_FAMILY,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import POLYNOMIAL_DECAY_LR, PolynomialDecayLRConfig
from fairseq2.recipes.common import (
    compile_model,
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    load_model,
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
from fairseq2.recipes.wav2vec2._common import Wav2Vec2Criterion, Wav2Vec2MetricBag
from fairseq2.recipes.wav2vec2._eval import Wav2Vec2EvalUnit
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class Wav2Vec2TrainConfig(TrainRecipeConfig):
    """Holds the configuration of a wav2vec 2.0 model training task.

    The default values correspond to the base ls960h training setup as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2", arch="base")
    )

    dataset: Wav2Vec2TrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2TrainDatasetSection()
    )

    trainer: Wav2Vec2TrainerSection = field(
        default_factory=lambda: Wav2Vec2TrainerSection(dtype=torch.float16)
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5e-04, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=POLYNOMIAL_DECAY_LR,
            config=PolynomialDecayLRConfig(num_warmup_steps=32_000),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=400_000,
            validate_every_n_steps=5_000,
            checkpoint_every_n_steps=25_000,
            publish_metrics_every_n_steps=200,
        )
    )


@dataclass(kw_only=True)
class Wav2Vec2TrainDatasetSection(DatasetSection):
    name: str | None = "librispeech_960h"
    """The name, path or path to the asset card of the speech dataset."""

    family: str = GENERIC_SPEECH_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"
    """The name of the train data split."""

    valid_split: str | None = "valid"
    """The name of the valid data split."""

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    example_shuffle_window: int = 0
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


@dataclass(kw_only=True)
class Wav2Vec2TrainerSection(TrainerSection):
    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


def register_wav2vec2_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Wav2Vec2TrainConfig)

    preset = registry.decorator

    @preset("base_960h")
    def base_960h() -> Wav2Vec2TrainConfig:
        config = Wav2Vec2TrainConfig()

        config.model.config = {"encoder_config": {"first_pass_dropout_p": 0.1}}

        return config

    @preset("large_960h")
    def large_960h() -> Wav2Vec2TrainConfig:
        config = Wav2Vec2TrainConfig()

        assert isinstance(config.optimizer.config, AdamWConfig)
        assert isinstance(config.lr_scheduler.config, PolynomialDecayLRConfig)

        config.model.arch = "large"
        config.model.config = {"encoder_config": {"first_pass_dropout_p": 0.1}}
        config.dataset.max_audio_len = 320_000
        config.dataset.max_num_elements = 1_200_000
        config.optimizer.config.lr = 3e-04
        config.lr_scheduler.config.num_warmup_steps = 20_000
        config.regime.num_steps = 250_000
        config.regime.publish_metrics_every_n_steps = 100

        return config


def load_wav2vec2_trainer(
    context: RuntimeContext, config: Wav2Vec2TrainConfig, output_dir: Path
) -> Trainer[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    dataset = load_dataset(SpeechDataset, context, config.dataset, gangs)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    log_model_config(log, config.model.config)

    model = load_model(Wav2Vec2Model, context, config, gangs, checkpoint_manager)

    dp_model = wrap_data_parallel(context, config, model, gangs, checkpoint_manager)

    dp_model = prepare_model(context, config, dp_model, gangs)

    log_model(log, dp_model, gangs)

    if config.trainer.torch_compile:
        model = compile_model(context, config.model, model)

    save_checkpoint_card(context, config, output_dir, gangs)

    optimizer = create_optimizer(context, config, dp_model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    # Initialize the train unit.
    criterion = Wav2Vec2Criterion(
        dp_model,
        config.trainer.diversity_loss_weight,
        config.trainer.feature_penalty_weight,
    )

    unit = Wav2Vec2TrainUnit(criterion, gangs)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = SpeechReadOptions(
        batching=batching,
        dtype=config.trainer.dtype,
        normalize_audio=config.dataset.normalize_audio,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        config.dataset.train_split,
        gangs.dp,
        config.dataset.min_audio_len,
        config.dataset.max_audio_len,
        read_options,
    )

    seed += 1

    # Initialize the validation unit.
    if config.dataset.valid_split is not None:
        valid_unit = Wav2Vec2EvalUnit(criterion, gangs)

        read_options = SpeechReadOptions(
            batching=batching,
            dtype=config.trainer.dtype,
            normalize_audio=config.dataset.normalize_audio,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
        )

        valid_data_reader = dataset.create_reader(
            config.dataset.valid_split,
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
class Wav2Vec2TrainUnit(AbstractTrainUnit[SequenceBatch]):
    _criterion: Wav2Vec2Criterion
    _metric_bag: Wav2Vec2MetricBag

    def __init__(self, criterion: Wav2Vec2Criterion, gangs: Gangs) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = Wav2Vec2MetricBag(gangs.dp)

    @override
    def __call__(self, batch: SequenceBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag

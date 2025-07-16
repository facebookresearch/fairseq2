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
from fairseq2.datasets import LengthBatching, SequenceBatch, SyncMode
from fairseq2.datasets.speech import (
    GENERIC_SPEECH_DATASET_FAMILY,
    SpeechDataset,
    SpeechReadOptions,
)
from fairseq2.device import CPU
from fairseq2.metrics import MetricBag
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import POLYNOMIAL_DECAY_LR, PolynomialDecayLRConfig
from fairseq2.recipes import Model, Trainer, TrainUnit
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_dataset,
    register_extra_asset_paths,
    setup_model,
    setup_torch,
    setup_training_gangs,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TrainerSection,
)
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

# isort: split

from fairseq2.recipes.wav2vec2._config import Wav2Vec2LossSection
from fairseq2.recipes.wav2vec2._criterion import Wav2Vec2Criterion
from fairseq2.recipes.wav2vec2._eval import Wav2Vec2EvalUnit


@dataclass(kw_only=True)
class Wav2Vec2TrainConfig:
    """
    The default values correspond to the base ls960h training setup as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2", arch="base")
    )

    dataset: Wav2Vec2TrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2TrainDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(dtype=torch.float16)
    )

    loss: Wav2Vec2LossSection = field(default_factory=lambda: Wav2Vec2LossSection())

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

    common: CommonSection = field(default_factory=lambda: CommonSection())


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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_wav2vec2_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(Wav2Vec2TrainConfig)

    preset = registry.decorator

    @preset("base_960h")
    def base_960h() -> Wav2Vec2TrainConfig:
        config = Wav2Vec2TrainConfig()

        config.model.config_overrides = {
            "encoder_config": {"first_pass_dropout_p": 0.1}
        }

        return config

    @preset("large_960h")
    def large_960h() -> Wav2Vec2TrainConfig:
        config = Wav2Vec2TrainConfig()

        assert isinstance(config.optimizer.config, AdamWConfig)
        assert isinstance(config.lr_scheduler.config, PolynomialDecayLRConfig)

        config.model.arch = "large"
        config.model.config_overrides = {
            "encoder_config": {"first_pass_dropout_p": 0.1}
        }
        config.dataset.max_audio_len = 320_000
        config.dataset.max_num_elements = 1_200_000
        config.optimizer.config.lr = 3e-04
        config.lr_scheduler.config.num_warmup_steps = 20_000
        config.regime.num_steps = 250_000
        config.regime.publish_metrics_every_n_steps = 100

        return config


def load_wav2vec2_trainer(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer:
    config = structure(config, Wav2Vec2TrainConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_training_gangs(context, config.gang, config.trainer)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_model(
        Wav2Vec2Model,
        context,
        config.model,
        config.trainer,
        output_dir,
        gangs,
        checkpoint_manager,
    )

    optimizer = create_optimizer(context, config.optimizer, model)

    lr_scheduler = create_lr_scheduler(
        context, config.lr_scheduler, config.regime, optimizer
    )

    dataset = load_dataset(SpeechDataset, context, config.dataset, gangs)

    # Initialize the train unit.
    criterion = Wav2Vec2Criterion(
        model.module, config.loss.diversity_weight, config.loss.features_penalty_weight
    )

    unit = Wav2Vec2TrainUnit(model, criterion)

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = SpeechReadOptions(
        batching=batching,
        dtype=config.trainer.dtype,
        normalize_audio=config.dataset.normalize_audio,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.grad_accumulation.num_batches,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
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
        valid_unit = Wav2Vec2EvalUnit(model, criterion)

        read_options = SpeechReadOptions(
            batching=batching,
            dtype=config.trainer.dtype,
            normalize_audio=config.dataset.normalize_audio,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
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
        config.trainer,
        config.regime,
        config.common,
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
        hyper_params=config,
    )


@final
class Wav2Vec2TrainUnit(TrainUnit[SequenceBatch]):
    _model: Model
    _criterion: Wav2Vec2Criterion

    def __init__(self, model: Model, criterion: Wav2Vec2Criterion) -> None:
        self._model = model

        self._criterion = criterion

    @override
    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._model

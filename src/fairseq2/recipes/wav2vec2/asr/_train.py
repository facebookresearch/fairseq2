# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, AsrDataset, AsrReadOptions
from fairseq2.gang import Gang, GangError
from fairseq2.logging import log
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import TRI_STAGE_LR, TriStageLRConfig
from fairseq2.recipes import Model, RecipeError, Trainer, TrainUnit
from fairseq2.recipes.asr import AsrCriterion, AsrEvalUnit, AsrMetricBag, AsrScorer
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_trainer,
    load_base_model,
    load_dataset,
    load_reference_model,
    load_text_tokenizer,
    prepare_model,
    register_extra_asset_paths,
    setup_data_parallel_model,
    setup_gangs,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    ReferenceModelSection,
    RegimeSection,
    TextTokenizerSection,
    TrainerSection,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2_asr", arch="base_10h")
    )

    pretrained_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )

    dataset: Wav2Vec2AsrTrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainDatasetSection()
    )

    text_tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="librispeech_asr")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

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
            validate_after_n_steps=10_000,
            validate_every_n_steps=1_000,
            publish_metrics_every_n_steps=200,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


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

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


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
        config.pretrained_model.name = "wav2vec2_large"
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
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer[Seq2SeqBatch]:
    config = structure(config, Wav2Vec2AsrTrainConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = load_base_model(
        Wav2Vec2AsrModel, context, config, output_dir, gangs, checkpoint_manager
    )

    module = cast(Wav2Vec2AsrModel, model.module)

    # If we start the training with an empty ASR model, use the weights of a
    # pretrained wav2vec 2.0 model.
    if model.is_empty_initialized:
        pt_model = load_reference_model(
            Wav2Vec2Model,
            context,
            config.pretrained_model.name,
            gangs,
            config.trainer.dtype,
            mp=config.trainer.mixed_precision is not None,
        )

        pt_module = cast(Wav2Vec2Model, pt_model.module)

        share_parameters(pt_module.encoder_frontend, module.encoder_frontend)
        share_parameters(pt_module.encoder, module.encoder)

        if module.masker is not None:
            share_parameters(pt_module.masker, module.masker)

        del pt_model

        # Make sure that the final projection layer is instantiated along with
        # the pretrained parameters if it was on the meta device.
        if gangs.dp.rank == 0:
            to_device(module, gangs.root.device)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the pretrained model load operation has failed. See the nested exception for details."
            ) from ex

    # We never train the feature extractor.
    freeze_parameters(module.encoder_frontend.feature_extractor)

    prepare_model(context, config, model, gangs)

    static_graph = config.trainer.freeze_encoder_for_n_steps == 0

    model = setup_data_parallel_model(context, config, model, gangs, static_graph)

    log_model(log, model.module, gangs)

    optimizer = create_optimizer(context, config, model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    dataset = load_dataset(AsrDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # Initialize the train unit.
    criterion = AsrCriterion(model)

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
        extras=config.dataset.extras,
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
        valid_scorer = AsrScorer(tokenizer)

        valid_criterion = AsrCriterion(model, valid_scorer)

        valid_unit = AsrEvalUnit(valid_criterion, gangs)

        read_options = AsrReadOptions(
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
        score_metric="wer",
    )


@final
class Wav2Vec2AsrTrainUnit(TrainUnit[Seq2SeqBatch]):
    _module: Wav2Vec2AsrModel
    _criterion: AsrCriterion
    _freeze_encoder_for_n_steps: int
    _metric_bag: AsrMetricBag

    def __init__(
        self, criterion: AsrCriterion, gang: Gang, freeze_encoder_for_n_steps: int
    ) -> None:
        """
        :param freeze_encoder_for_n_steps: The encoder will be frozen for this
            number of steps.
        """
        module = criterion.model.base_module

        if not isinstance(module, Wav2Vec2AsrModel):
            raise TypeError(
                f"`criterion.model.base_module` must be of type `{Wav2Vec2AsrModel}`, but is of type `{type(module)}` instead."
            )

        self._module = module
        self._criterion = criterion
        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps

        self._metric_bag = AsrMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @override
    def set_step_nr(self, step_nr: int) -> None:
        module = self._module

        if step_nr <= self._freeze_encoder_for_n_steps:
            if step_nr == 1:
                log.info("Freezing the encoder for the first {} steps.", self._freeze_encoder_for_n_steps)  # fmt: skip

            freeze_parameters(module.encoder_frontend)
            freeze_parameters(module.encoder)

            if module.masker is not None:
                freeze_parameters(module.masker)
        else:
            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info("Unfreezing the encoder after step {}.", step_nr - 1)

            freeze_parameters(module, False)

            # We never train the feature extractor.
            freeze_parameters(module.encoder_frontend.feature_extractor)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> AsrMetricBag:
        return self._metric_bag

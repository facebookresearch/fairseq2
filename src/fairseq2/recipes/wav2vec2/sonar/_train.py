# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final, Literal

import torch

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.sonarspeech import (
    GENERIC_SONAR_SPEECH_DATASET_FAMILY,
    GenericSonarSpeechDataset,
)
from fairseq2.datasets.speech import ManifestDatasetInterface, SpeechReadOptions
from fairseq2.gang import Gang, GangError
from fairseq2.logging import log
from fairseq2.models.asr import AsrModel
from fairseq2.models.llama.lora import get_llama_lora_config
from fairseq2.models.seq2seq import Seq2SeqBatch, SonarSpeechSeq2SeqBatch
from fairseq2.models.transformer_lm import TransformerLanguageModel
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.sonar._model import SonarSpeechEncoderModel
from fairseq2.nn.lora import wrap_lora
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import (
    COSINE_ANNEALING_LR,
    CosineAnnealingLRConfig,
    MYLE_LR,
    MyleLRConfig,
    TRI_STAGE_LR,
    TriStageLRConfig,
)
from fairseq2.recipes import Model, RecipeError, Trainer, TrainUnit
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
    ReferenceModelSection,
    RegimeSection,
    TextTokenizerSection,
    TrainerSection,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.wav2vec2.asr._train import Wav2Vec2AsrTrainConfig
from fairseq2.recipes.wav2vec2.batch_weighted_datareader import (
    BatchMixtureDataset,
    MIXTURE_DATASET_FAMILY,
)
from fairseq2.recipes.wav2vec2.sonar._criterion import (
    SonarSpeechCriterion,
    SonarSpeechMetricBag,
)
from fairseq2.recipes.wav2vec2.sonar._eval import SonarSpeechEvalUnit
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate
from torch import Tensor
from typing_extensions import override


@dataclass(kw_only=True)
class SonarSpeechTrainConfig:
    """Holds the configuration of a Sonar2 training task."""

    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="wav2vec2_sonar_speech", arch="7b_fleurs"
        )
    )

    pretrained_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )

    dataset: SonarSpeechTrainDatasetSection = field(
        default_factory=lambda: SonarSpeechTrainDatasetSection()
    )

    tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="librispeech_asr")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: SonarSpeechTrainerSection = field(
        default_factory=lambda: SonarSpeechTrainerSection(
            dtype=torch.float16, gradient_accumulation=1
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER, config=AdamWConfig(lr=5e-05, betas=(0.9, 0.98))
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=MYLE_LR, config=MyleLRConfig(start_lr=1e-7, num_warmup_steps=8_000)
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=16_800,
            validate_every_n_steps=16_800,
            checkpoint_every_n_steps=16_800,
            publish_metrics_every_n_steps=10,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class SonarSpeechTrainDatasetSection(DatasetSection):
    name: str | None = "librilight_asr_10h"

    family: str = GENERIC_SONAR_SPEECH_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"

    valid_split: str | None = "dev"

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

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    no_padding: bool = False

    # Upsampling
    beta_corpus: float | None = None
    beta_language: float | None = None
    """Params specifying sampling temperature; between [0,1]."""

    # SpecAugment
    spec_aug_p: float | None = None
    """Probability of applying SpecAugment per row."""
    spec_aug_freq_mask_param: int = 80
    """Maximum frequency mask length."""
    spec_aug_time_mask_param: int = 80
    """Maximum time mask length."""

    always_read_tsv: bool = False
    """If ``True``, always read the TSV manifest, regardless of whether parquet datasets exist."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


@dataclass(kw_only=True)
class SonarSpeechTrainerSection(TrainerSection):
    freeze_encoder_for_n_steps: int = 0
    """ """


def register_sonar_speech_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarSpeechEncoderConfig)

    preset = registry.decorator

    w2v2_encoder_registry = context.get_config_registry(Wav2Vec2EncoderConfig)

    @preset("base_10h")
    def base_10h() -> SonarSpeechEncoderConfig:

        return SonarSpeechEncoderConfig()

    @preset("7b_fleurs")
    def fleurs_7b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        return config

    @preset("1b_fleurs")
    def fleurs_1b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("1b")

        return config

    @preset("7b_fleurs_mean")
    def fleurs_7b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")
        config.pooling_type = "mean"

        return config


def load_sonar_speech_trainer(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer[SonarSpeechSeq2SeqBatch]:
    config = structure(config, SonarSpeechTrainConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_training_gangs(context, config.gang, config.trainer)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = load_base_model(
        SonarSpeechEncoderModel,
        context,
        config.model,
        config.trainer,
        output_dir,
        gangs,
        checkpoint_manager,
    )

    module = cast(SonarSpeechEncoderModel, model.module)

    # If we start the training with an empty ASR model, use the weights of a
    # pretrained wav2vec 2.0 model.
    if model.is_empty_initialized:
        pt_model = load_reference_model(
            Wav2Vec2Model,
            context,
            config.pretrained_model,
            gangs,
            config.trainer.dtype,
            mp=config.trainer.mixed_precision != "off",
        )

        pt_module = cast(Wav2Vec2Model, pt_model.module)

        share_parameters(pt_module.encoder_frontend, module.encoder_frontend)  # type: ignore
        share_parameters(pt_module.encoder, module.encoder)  # type: ignore

        if module.masker is not None:
            share_parameters(pt_module.masker, module.masker)  # type: ignore

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

    freeze_parameters(module.encoder_frontend.feature_extractor)  # type: ignore

    prepare_model(context, config.trainer, model)

    static_graph = config.trainer.freeze_encoder_for_n_steps == 0

    model = setup_data_parallel_model(
        context, config.trainer, model, gangs, static_graph
    )

    log_model(log, model.module, gangs)

    optimizer = create_optimizer(context, config.optimizer, model)

    lr_scheduler = create_lr_scheduler(
        context, config.lr_scheduler, config.regime, optimizer
    )

    tokenizer = load_text_tokenizer(context, config.tokenizer)

    dataset = load_dataset(GenericSonarSpeechDataset, context, config.dataset, gangs)

    # Initialize the train unit.
    criterion = SonarSpeechCriterion(model)

    unit = SonarSpeechTrainUnit(
        criterion, gangs, config.trainer.freeze_encoder_for_n_steps
    )

    batching = LengthBatching(config.dataset.max_num_elements)

    read_options = SpeechReadOptions(
        batching=batching,
        dtype=config.trainer.dtype,
        normalize_audio=config.dataset.normalize_audio,
        example_shuffle_window=config.dataset.example_shuffle_window,
        batch_shuffle_window=config.dataset.batch_shuffle_window,
        num_accumulate=config.trainer.gradient_accumulation,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
        npc=config.dataset.npc,
        beta_corpus=config.dataset.beta_corpus,
        beta_language=config.dataset.beta_language,
        spec_aug_p=config.dataset.spec_aug_p,
        spec_aug_freq_mask_param=config.dataset.spec_aug_freq_mask_param,
        spec_aug_time_mask_param=config.dataset.spec_aug_time_mask_param,
    )

    data_reader = dataset.create_reader(
        config.dataset.train_split,
        tokenizer,
        gangs.dp,
        min_audio_len=config.dataset.min_audio_len,
        max_audio_len=config.dataset.max_audio_len,
        options=read_options,
    )

    seed += 1

    # Initialize the validation unit.
    if config.dataset.valid_split is not None:

        read_options = SpeechReadOptions(
            batching=batching,
            dtype=config.trainer.dtype,
            normalize_audio=config.dataset.normalize_audio,
            example_shuffle_window=1,
            batch_shuffle_window=1,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        valid_units = []
        valid_data_readers = []
        valid_splits = [s.strip() for s in (config.dataset.valid_split).split(",")]
        for i in range(len(valid_splits)):
            valid_unit = SonarSpeechEvalUnit(criterion, gangs)
            valid_units.append(valid_unit)

            valid_data_reader = dataset.create_reader(
                valid_splits[i],
                tokenizer,
                gangs.dp,
                min_audio_len=config.dataset.min_audio_len,
                max_audio_len=config.dataset.max_audio_len,
                options=read_options,
            )
            valid_data_readers.append(valid_data_reader)
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
    )


@final
class SonarSpeechTrainUnit(TrainUnit[SonarSpeechSeq2SeqBatch]):
    _module: SonarSpeechEncoderModel
    _criterion: SonarSpeechCriterion
    _metric_bag: SonarSpeechMetricBag
    _freeze_encoder_for_n_steps: int
    _frozen: bool

    def __init__(
        self,
        criterion: SonarSpeechCriterion,
        gangs: Gangs,
        freeze_encoder_for_n_steps: int,
    ) -> None:
        module = criterion.model.base_module

        if not isinstance(module, SonarSpeechEncoderModel):
            raise TypeError(
                f"`criterion.model.base_module` must be of type `{SonarSpeechEncoderModel}`, but is of type `{type(module)}` instead."
            )

        self._module = module
        self._criterion = criterion
        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps
        self._frozen = False
        self._metric_bag = SonarSpeechMetricBag(gangs.dp)

    @override
    def __call__(self, batch: SonarSpeechSeq2SeqBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @override
    def set_step_nr(self, step_nr: int) -> None:
        module = self._module

        if step_nr <= self._freeze_encoder_for_n_steps:
            if self._frozen:
                return

            if step_nr == 1:
                log.info("Freezing the encoder for the first {} steps.", self._freeze_encoder_for_n_steps)  # fmt: skip

            freeze_parameters(module.encoder_frontend)  # type: ignore
            freeze_parameters(module.encoder)  # type: ignore

            if module.masker is not None:
                freeze_parameters(module.masker)  # type: ignore

            self._frozen = True
        else:
            if not self._frozen:
                return

            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info("Unfreezing the encoder after step {}.", step_nr - 1)

            freeze_parameters(module, False)

            # We never train the feature extractor.
            freeze_parameters(module.encoder_frontend.feature_extractor)  # type: ignore

            self._frozen = False

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> SonarSpeechMetricBag:
        return self._metric_bag

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.datasets import LengthBatching, SyncMode
from fairseq2.datasets.asr import GENERIC_ASR_DATASET_FAMILY, AsrDataset
from fairseq2.datasets.speech import ManifestDatasetInterface, SpeechReadOptions
from fairseq2.gang import Gang, GangError
from fairseq2.logging import log
from fairseq2.models.asr import AsrModel
from fairseq2.models.llama.lora import get_llama_lora_config
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.transformer_lm import TransformerLanguageModel
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.nn.lora import wrap_lora
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
from fairseq2.recipes.wav2vec2.batch_weighted_datareader import (
    MIXTURE_DATASET_FAMILY,
    BatchMixtureDataset,
)
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


def _strict_name(s: str) -> str:
    """
    Maps a string to a strict version containing only
    alphanumeric characters, dash, underscore, and forward slash.
    """
    # Use regex to keep only allowed characters
    return re.sub(r"[^a-zA-Z0-9\-_\/]", "_", s)


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2_asr", arch="base_10h")
    )

    pretrained_encoder: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )
    pretrained_encoder_is_ctc: bool = False
    pretrained_model_full: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="")
    )
    pretrained_decoder: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="")
    )

    dataset: Wav2Vec2AsrTrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainDatasetSection()
    )

    tokenizer: TextTokenizerSection = field(
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

    best_checkpoint_metric: Literal["uer", "wer"] = "uer"


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

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    always_read_tsv: bool = False
    """If ``True``, always read the TSV manifest, regardless of whether parquet datasets exist."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""

    max_num_batches: int | None = None
    """The maximum number of batches for the dataloader to return."""

    max_batch_size: int = -1
    """The maximum batch size (num examples). If ``-1``, no maximum is applied."""

    min_samples_per_char: int = 160
    """If a sample has more than ``sample_rate / min_samples_per_char`` chars per second, it's filtered out."""

    wer_logging: bool = False
    """If ``True``, log WER scores when validating."""

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

    # Zero/Few-shot
    n_context_examples: int = 0
    """The number of context examples to use when providing context."""
    bucket_size_train: int = 2000
    """Minimum size of pool for choosing context examples, for training set."""
    bucket_size_eval: int = 30
    """Minimum size of pool for choosing context examples, for eval sets."""
    batch_with_context_length: bool = True
    """Use total batch of speech + context speech for length batching."""


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
        config.pretrained_encoder.name = "wav2vec2_large"
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

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_training_gangs(context, config.gang, config.trainer)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = load_base_model(
        AsrModel,
        context,
        config.model,
        config.trainer,
        output_dir,
        gangs,
        checkpoint_manager,
    )

    module = cast(AsrModel, model.module)

    # If we start the training with an empty ASR model, use the weights of a
    # pretrained wav2vec 2.0 model.
    if model.is_empty_initialized and config.pretrained_encoder.name:
        tp = AsrModel if config.pretrained_encoder_is_ctc else Wav2Vec2Model
        pt_model = load_reference_model(
            tp,
            context,
            config.pretrained_encoder,
            gangs,
            config.trainer.dtype,
            mp=config.trainer.mixed_precision != "off",
        )

        pt_module = cast(tp, pt_model.module)  # type: ignore

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
            log.info("Pretrained encoder loaded")
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the pretrained model load operation has failed. See the nested exception for details."
            ) from ex

    # Load a full model from checkpoint
    if config.pretrained_model_full.name:
        assert not config.pretrained_encoder.name

        pt_model = load_reference_model(
            type(model.module),
            context,
            config.pretrained_model_full,
            gangs,
            config.trainer.dtype,
            mp=config.trainer.mixed_precision != "off",
        )
        pt_module = pt_model.module  # type: ignore[assignment]
        share_parameters(pt_module, module)
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

    # Load a pretrained decoder model from checkpoint
    if config.pretrained_decoder.name:
        assert not config.pretrained_model_full.name

        pt_model = load_reference_model(
            TransformerLanguageModel,
            context,
            config.pretrained_decoder,
            gangs,
            config.trainer.dtype,
            mp=config.trainer.mixed_precision != "off",
        )
        pt_module = pt_model.module  # type: ignore[assignment]
        share_parameters(pt_module.decoder, module.llama_decoder)  # type: ignore
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

        # LoRA adapters
        module = wrap_lora(module, get_llama_lora_config())  # type: ignore[assignment]

    # We never train the feature extractor.
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

    # Initialize the train unit.
    criterion = AsrCriterion(model)

    unit = Wav2Vec2AsrTrainUnit(
        criterion, gangs.dp, config.trainer.freeze_encoder_for_n_steps
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
        max_num_batches=config.dataset.max_num_batches,
        n_context_examples=config.dataset.n_context_examples,
        bucket_size=config.dataset.bucket_size_train,
        deterministic_context=False,
        max_batch_size=config.dataset.max_batch_size,
        min_samples_per_char=config.dataset.min_samples_per_char,
        batch_with_context_length=config.dataset.batch_with_context_length,
    )

    dataset: AsrDataset | BatchMixtureDataset
    if config.dataset.family == MIXTURE_DATASET_FAMILY:
        log.info(
            f"Loading dataset {config.dataset.name} with family={config.dataset.family}"
        )
        dataset = BatchMixtureDataset.from_configs(
            AsrDataset, context, config.dataset, gangs
        )
    else:
        dataset = load_dataset(AsrDataset, context, config.dataset, gangs)
        if (
            isinstance(dataset, ManifestDatasetInterface)
            and config.dataset.always_read_tsv
        ):
            dataset._always_read_tsv = True

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
        eval_read_options = deepcopy(read_options)
        valid_scorer = AsrScorer(tokenizer, verbose=config.dataset.wer_logging)

        valid_criterion = AsrCriterion(model, valid_scorer)

        eval_read_options = SpeechReadOptions(
            batching=batching,
            dtype=config.trainer.dtype,
            normalize_audio=config.dataset.normalize_audio,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            max_num_batches=config.dataset.max_num_batches,
            npc=config.dataset.npc,
            extras=deepcopy(config.dataset.extras),
            n_context_examples=config.dataset.n_context_examples,
            bucket_size=config.dataset.bucket_size_eval,
            deterministic_context=True,
            max_batch_size=config.dataset.max_batch_size,
            min_samples_per_char=config.dataset.min_samples_per_char,
            batch_with_context_length=config.dataset.batch_with_context_length,
        )

        # replace partition_filters from read_options
        eval_read_options.extras["partition_filters"] = eval_read_options.extras.get(
            "eval_partition_filters"
        )

        valid_units = []
        valid_data_readers = []
        if config.dataset.family == MIXTURE_DATASET_FAMILY:
            # config.dataset.valid_split = "dataset0=[dev,test],dataset1=[validation,test]"
            assert isinstance(dataset, BatchMixtureDataset)
            valid_splits = dataset.parse_split_config_with_multiple_splits(
                config.dataset.valid_split
            )
            # ["dataset0=dev", "dataset0=test", "dataset1=validation", "dataset1=test"]
        else:
            # config.dataset.valid_split = "dev,test"
            valid_splits = [s.strip() for s in (config.dataset.valid_split).split(",")]

        readers_partition_columns = config.dataset.extras.get(
            "readers_partition_columns", None
        )

        if readers_partition_columns:
            for single_vsplit in valid_splits:
                # currently implemented only for mixture_lance_asr family
                assert hasattr(dataset, "create_multi_readers")
                multi_readers = dataset.create_multi_readers(  # type: ignore
                    readers_partition_columns=readers_partition_columns,
                    split=single_vsplit,
                    tokenizer=tokenizer,
                    gang=gangs.dp,
                    min_audio_len=config.dataset.min_audio_len,
                    max_audio_len=config.dataset.max_audio_len,
                    options=deepcopy(eval_read_options),
                )
                for name, valid_data_reader in multi_readers.items():
                    name = single_vsplit + "_" + name
                    name = _strict_name(name)
                    valid_unit = AsrEvalUnit(valid_criterion, gangs, name)
                    valid_units.append(valid_unit)
                    valid_data_readers.append(valid_data_reader)
        else:
            for single_vsplit in valid_splits:
                name = single_vsplit.replace("=", "_")
                valid_unit = AsrEvalUnit(valid_criterion, gangs, name)
                valid_units.append(valid_unit)

                valid_data_reader = dataset.create_reader(
                    single_vsplit,
                    tokenizer,
                    gangs.dp,
                    config.dataset.min_audio_len,
                    config.dataset.max_audio_len,
                    deepcopy(eval_read_options),
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
        score_metric=config.best_checkpoint_metric,
    )


@final
class Wav2Vec2AsrTrainUnit(TrainUnit[Seq2SeqBatch]):
    _module: AsrModel
    _criterion: AsrCriterion
    _freeze_encoder_for_n_steps: int
    _frozen: bool
    _metric_bag: AsrMetricBag

    def __init__(
        self, criterion: AsrCriterion, gang: Gang, freeze_encoder_for_n_steps: int
    ) -> None:
        """
        :param freeze_encoder_for_n_steps: The encoder will be frozen for this
            number of steps.
        """
        module = criterion.model.base_module

        if not isinstance(module, AsrModel):
            raise TypeError(
                f"`criterion.model.base_module` must be of type `{AsrModel}`, but is of type `{type(module)}` instead."
            )

        self._module = module
        self._criterion = criterion
        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps
        self._frozen = False

        self._metric_bag = AsrMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
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
    def metric_bag(self) -> AsrMetricBag:
        return self._metric_bag

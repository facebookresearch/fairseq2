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
from fairseq2.datasets import Batching, LengthBatching, StaticBatching, SyncMode
from fairseq2.datasets.parallel_text import (
    GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
    ParallelTextDataset,
    ParallelTextReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.generation import BeamSearchConfig
from fairseq2.metrics.text import DEFAULT_BLEU_TOKENIZER
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import MYLE_LR, MyleLRConfig
from fairseq2.recipes import EvalUnit, Model, Seq2SeqMetricBag, Trainer, TrainUnit
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    create_lr_scheduler,
    create_optimizer,
    create_seq2seq_generator,
    create_trainer,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_model,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    FsdpSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    Seq2SeqGeneratorSection,
    TextTokenizerSection,
    TrainerSection,
)
from fairseq2.recipes.mt._common import MTCriterion, MTLossSection
from fairseq2.recipes.mt._eval import MTBleuChrfEvalUnit, MTLossEvalUnit
from fairseq2.typing import CPU
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class MTTrainConfig:
    """
    The default values correspond to the baseline NLLB-200 training setup as
    described in cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="transformer", arch="nllb_dense_600m"
        )
    )

    dataset: MTTrainDatasetSection = field(
        default_factory=lambda: MTTrainDatasetSection()
    )

    text_tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="nllb-200")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.float16,
            data_parallelism="fsdp",
            fsdp=FsdpSection(granularity="stack"),
            gradient_accumulation=2,
        )
    )

    loss: MTLossSection = field(default_factory=lambda: MTLossSection())

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER, config=AdamWConfig(lr=0.001, betas=(0.9, 0.98))
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=MYLE_LR, config=MyleLRConfig(start_lr=1e-7, num_warmup_steps=8_000)
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=100_000,
            validate_every_n_steps=10_000,
            checkpoint_every_n_steps=10_000,
            publish_metrics_every_n_steps=200,
        )
    )

    validation: MTValidationSection = field(
        default_factory=lambda: MTValidationSection()
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class MTTrainDatasetSection(DatasetSection):
    name: str | None = "foo"  # TODO: change!

    family: str = GENERIC_PARALLEL_TEXT_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"

    valid_split: str | None = "valid"

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    max_num_tokens: int = 4096
    """The maximum number of tokens per batch."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


@dataclass(kw_only=True)
class MTValidationSection:
    compute_bleu_chrf: bool = True
    """If ``True``, computes BLEU and chrF++ during validation."""

    bleu_tokenizer = DEFAULT_BLEU_TOKENIZER
    """The tokenizer to compute the BLEU metric."""

    seq2seq_generator: Seq2SeqGeneratorSection = field(
        default_factory=lambda: Seq2SeqGeneratorSection(
            config=BeamSearchConfig(max_gen_len=(1, 256), echo_prompt=True),
            batch_size=8,
        )
    )


def register_mt_train_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(MTTrainConfig)

    preset = registry.decorator

    @preset("nllb_dense_300m")
    def nllb_dense_300m() -> MTTrainConfig:
        config = nllb_dense()

        assert isinstance(config.lr_scheduler.config, MyleLRConfig)

        config.model.arch = "nllb_dense_300m"
        config.trainer.gradient_accumulation = 4
        config.lr_scheduler.config.num_warmup_steps = 400
        config.regime.num_steps = 10_000
        config.regime.validate_every_n_steps = 1000
        config.regime.checkpoint_every_n_steps = 1000

        return config

    @preset("nllb_dense")
    def nllb_dense() -> MTTrainConfig:
        return MTTrainConfig()


def load_mt_trainer(
    context: RuntimeContext, config: object, output_dir: Path
) -> Trainer[Seq2SeqBatch]:
    config = structure(config, MTTrainConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_model(
        EncoderDecoderModel, context, config, output_dir, gangs, checkpoint_manager
    )

    optimizer = create_optimizer(context, config, model)

    lr_scheduler = create_lr_scheduler(context, config, optimizer)

    dataset = load_dataset(ParallelTextDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # Initialize the train unit.
    criterion = MTCriterion(model, label_smoothing=config.loss.label_smoothing)

    unit = MTTrainUnit(criterion, gangs)

    batching: Batching = LengthBatching(config.dataset.max_num_tokens)

    read_options = ParallelTextReadOptions(
        batching=batching,
        sample=True,
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
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    train_seed = seed

    seed += 1

    # Initialize the validation units.
    if config.validation.compute_bleu_chrf:
        seq2seq_generator = create_seq2seq_generator(context, config.validation, model)
    else:
        seq2seq_generator = None

    valid_units: list[EvalUnit[Seq2SeqBatch]] = []

    valid_data_readers = []

    valid_split = config.dataset.valid_split

    if valid_split is None:
        directions = []
    else:
        directions = dataset.directions(valid_split)

    for direction in directions:
        assert valid_split is not None

        valid_loss_unit = MTLossEvalUnit(criterion, direction, gangs)

        valid_units.append(valid_loss_unit)

        batching = LengthBatching(config.dataset.max_num_tokens)

        read_options = ParallelTextReadOptions(
            direction=direction,
            batching=batching,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        valid_data_reader = dataset.create_reader(
            valid_split,
            tokenizer,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        seed += 1

        valid_data_readers.append(valid_data_reader)

        if config.validation.compute_bleu_chrf:
            assert seq2seq_generator is not None

            valid_score_unit = MTBleuChrfEvalUnit(
                model,
                direction,
                seq2seq_generator,
                tokenizer,
                gangs,
                bleu_tokenizer=config.validation.bleu_tokenizer,
            )

            valid_units.append(valid_score_unit)

            batching = StaticBatching(config.validation.seq2seq_generator.batch_size)

            options = ParallelTextReadOptions(
                direction=direction,
                batching=batching,
                sync_mode=SyncMode.UNTIL_LAST,
                num_prefetch=config.dataset.num_prefetch,
                seed=seed,
                extras=config.dataset.extras,
            )

            valid_data_reader = dataset.create_reader(
                valid_split,
                tokenizer,
                gangs.dp,
                config.dataset.min_seq_len,
                config.dataset.max_seq_len,
                options,
            )

            valid_data_readers.append(valid_data_reader)

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
        train_seed,
    )


@final
class MTTrainUnit(TrainUnit[Seq2SeqBatch]):
    _criterion: MTCriterion
    _metric_bag: Seq2SeqMetricBag

    def __init__(self, criterion: MTCriterion, gangs: Gangs) -> None:
        self._criterion = criterion

        self._metric_bag = Seq2SeqMetricBag(gangs.dp)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> Seq2SeqMetricBag:
        return self._metric_bag

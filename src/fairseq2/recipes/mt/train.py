# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider, FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets import LengthBatching, StaticBatching
from fairseq2.datasets.parallel_text import (
    GenericParallelTextDataset,
    load_parallel_text_dataset,
)
from fairseq2.gang import Gang
from fairseq2.generation.encoder_decoder import BeamSearchConfig, generator_factories
from fairseq2.logging import get_log_writer
from fairseq2.models import model_factories
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqBatch, as_auto_regressive_input
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer import transformer_archs
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.recipes.common_metrics import Seq2SeqMetricBag
from fairseq2.recipes.evaluator import EvalUnit
from fairseq2.recipes.mt.eval import MTBleuChrfEvalUnit, MTLossEvalUnit
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.recipes.utils.setup import (
    check_model_type,
    setup_root_gang,
    to_data_parallel,
)
from fairseq2.typing import META, DataClass, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class MTTrainConfig:
    """Holds the configuration of a machine translation training task.

    The default values correspond to the baseline NLLB-200 training setup as
    described in cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.
    """

    # Data
    dataset: AssetReference = "foo"  # TODO: change!
    """The name, path, or path to the asset card of the parallel text dataset."""

    split: str = "train"
    """The name of the train data split."""

    valid_split: str = "valid"
    """The name of the valid data split."""

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

    tokenizer: AssetReference = "nllb-200"
    """The name or path to the asset card of the tokenizer to use."""

    # Model
    model_family: str = "transformer"
    """The family of the model."""

    model_arch: Optional[str] = "nllb_dense_600m"
    """The architecture of the model."""

    model_config: Optional[DataClass] = field(
        default_factory=lambda: transformer_archs.get(
            "nllb_dense_600m", return_empty=True
        )
    )
    """The configuration of the model."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "fsdp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "stack"
    """The granularity at which to wrap the ASR model."""

    # Optimizer, LR, and Loss
    lr: float = 0.001
    """The initial (post-warm-up) learning rate."""

    start_lr: float = 1e-7
    """The initial warm-up learning rate."""

    num_lr_warmup_steps: int = 8000
    """The number of learning rate warm-up steps."""

    betas: tuple[float, float] = (0.9, 0.98)
    """The coefficients of AdamW."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gradient_accumulation: int = 2
    """The number of steps to accumulate gradients before an optimizer update."""

    label_smoothing: float = 0.1
    """The amount of label smoothing to apply while computing the loss."""

    # Regime
    max_num_steps: int = 100_000
    """The maximum number of steps to train for."""

    max_num_data_epochs: Optional[int] = None
    """The maximum number of data epochs to train for."""

    validate_after_n_steps: int = 0
    """The number of steps after which to start validating the model."""

    validate_every_n_steps: int = 10_000
    """The step interval at which to validate the model."""

    checkpoint_after_n_steps: int = 0
    """The number of steps after which to start checkpointing."""

    checkpoint_every_n_steps: int = 10_000
    """The step interval at which to checkpoint."""

    publish_metrics_every_n_steps: int = 200
    """The step interval at which to publish metrics."""

    # Checkpointing
    resume_checkpoint_dir: Optional[Path] = None
    """If not ``None``, adds the specified path to the default asset store."""

    # BLEU/chrF++
    compute_bleu_chrf: bool = True
    """If ``True``, computes BLEU and chrF++ during validation."""

    generator: str = "beam_search"
    """The sequence generator."""

    generator_config: Optional[DataClass] = field(
        default_factory=lambda: BeamSearchConfig()
    )
    """The configuration of the sequence generator."""

    generator_batch_size: int = 8
    """The number of sentences per batch."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: Optional[tuple[int, int]] = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, enables the anomaly detection feature of ``torch.autograd``."""


mt_train_presets = ConfigRegistry[MTTrainConfig]()

mt_train_preset = mt_train_presets.decorator


@mt_train_preset("nllb_dense_300m")
def _nllb_dense_300m() -> MTTrainConfig:
    model_config = transformer_archs.get("nllb_dense_300m", return_empty=True)

    config = _nllb_dense_600m()

    config.model_arch = "nllb_dense_300m"
    config.model_config = model_config
    config.num_lr_warmup_steps = 400
    config.gradient_accumulation = 4
    config.max_num_steps = 10_000
    config.validate_every_n_steps = 1000
    config.checkpoint_every_n_steps = 1000

    return config


@mt_train_preset("nllb_dense_600m")
def _nllb_dense_600m() -> MTTrainConfig:
    return MTTrainConfig()


def load_mt_trainer(config: MTTrainConfig, output_dir: Path) -> Trainer[Seq2SeqBatch]:
    """Load a :class:`Trainer` for machine translation training."""
    wall_watch = Stopwatch(start=True)

    gang = setup_root_gang(log, monitored=config.monitored_gang)

    checkpoint_manager = FileCheckpointManager(output_dir.joinpath("checkpoints"), gang)

    if config.resume_checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.resume_checkpoint_dir)
        )

    seed = config.seed

    tokenizer_card = retrieve_asset_card(config.tokenizer)

    # Load the tokenizer.
    log.info("Loading {} tokenizer.", tokenizer_card.name)

    tokenizer = load_text_tokenizer(tokenizer_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} parallel text dataset.", dataset_card.name)

        dataset = load_parallel_text_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericParallelTextDataset.from_path(dataset_path)

    # Initialize the model
    try:
        model_factory = model_factories.get(
            config.model_family, config.model_config, config.model_arch
        )

        model = model_factory(device=META, dtype=torch.float32)
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, EncoderDecoderModel):
        raise ValueError(
            f"The model must be of type `{EncoderDecoderModel}`, but is of type `{type(model)}` instead."
        )

    log_model_config(model_factory.config, log)

    checkpoint_manager.save_model_metadata(
        family=model.family,
        config=model_factory.config,
        tokenizer_name=tokenizer_card.name,
    )

    has_checkpoint = checkpoint_manager.has_checkpoint()

    dp_model = to_data_parallel(
        model,
        gang,
        config.data_parallelism,
        log,
        fsdp_skip_init=has_checkpoint,
        fsdp_broadcast_state=not has_checkpoint,
        fsdp_mixed_precision_dtype=config.dtype,
        fsdp_fp32_reduce=True,
        fsdp_wrap_granularity=config.fsdp_wrap_granularity,
    )

    log_model(dp_model, log, rank=gang.rank)

    # Initialize the train unit and the optimizer.
    unit = MTTrainUnit(dp_model, gang, label_smoothing=config.label_smoothing)

    data_reader = dataset.create_reader(
        config.split,
        tokenizer,
        gang,
        config.max_seq_len,
        batching=LengthBatching(config.max_num_tokens),
        sample=True,
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    seed += 1

    optimizer = AdamW(dp_model.parameters(), lr=config.lr, betas=config.betas)

    lr_scheduler = MyleLR(
        optimizer,
        num_warmup_steps=config.num_lr_warmup_steps,
        start_lr=config.start_lr,
    )

    # Initialize the sequence generator.
    if config.compute_bleu_chrf:
        try:
            generator_factory = generator_factories.get(
                config.generator, config.generator_config
            )

            generator = generator_factory(model)
        except ValueError as ex:
            raise ValueError(
                "The sequence generator cannot be created. See nested exception for details."
            ) from ex
    else:
        generator = None

    # Initialize the validation units.
    valid_units: list[EvalUnit[Seq2SeqBatch]] = []

    valid_data_readers = []

    for direction in dataset.directions(config.valid_split):
        # Loss Validation
        valid_loss_unit = MTLossEvalUnit(
            dp_model,
            direction,
            gang,
            label_smoothing=config.label_smoothing,
        )

        valid_units.append(valid_loss_unit)

        valid_data_reader = dataset.create_reader(
            config.valid_split,
            tokenizer,
            gang,
            config.max_seq_len,
            batching=LengthBatching(config.max_num_tokens),
            direction=direction,
            sync_batches=False,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )

        seed += 1

        valid_data_readers.append(valid_data_reader)

        if config.compute_bleu_chrf:
            assert generator is not None

            # BLEU/chrF++ Validation
            valid_score_unit = MTBleuChrfEvalUnit(direction, generator, tokenizer, gang)

            valid_units.append(valid_score_unit)

            valid_data_reader = dataset.create_reader(
                config.valid_split,
                tokenizer,
                gang,
                config.max_seq_len,
                batching=StaticBatching(config.generator_batch_size),
                direction=direction,
                sync_batches=False,
                num_prefetch=config.num_prefetch,
                seed=seed,
            )

            seed += 1

            valid_data_readers.append(valid_data_reader)

    # Initialize the trainer.
    return Trainer[Seq2SeqBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=gang,
        dtype=config.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=config.fp16_loss_scale,
        max_gradient_norm=config.max_gradient_norm,
        max_num_steps=config.max_num_steps,
        max_num_data_epochs=config.max_num_data_epochs,
        score_metric_name="chrf" if config.compute_bleu_chrf else None,
        valid_units=valid_units,
        valid_data_readers=valid_data_readers,
        validate_after_n_steps=config.validate_after_n_steps,
        validate_every_n_steps=config.validate_every_n_steps,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=config.checkpoint_after_n_steps,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class MTTrainUnit(AbstractTrainUnit[Seq2SeqBatch]):
    """Represents a machine translation training unit."""

    _label_smoothing: float
    _metric_bag: Seq2SeqMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        *,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(model)

        check_model_type(model, EncoderDecoderModel)

        self._label_smoothing = label_smoothing

        self._metric_bag = Seq2SeqMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
        input_batch, target_batch = as_auto_regressive_input(batch)

        output = self._forward(input_batch)

        loss = output.compute_loss(
            target_batch.seqs, label_smoothing=self._label_smoothing
        )

        self._metric_bag.update_nll_loss(input_batch, loss.detach())

        self._metric_bag.update_batch_metrics(input_batch)

        return loss, batch.num_target_elements()

    def _forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> Seq2SeqMetricBag:
        return self._metric_bag

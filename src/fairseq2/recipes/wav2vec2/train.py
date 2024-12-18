# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider, FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.datasets.batching import LengthBatching
from fairseq2.datasets.speech import GenericSpeechDataset, load_speech_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import create_model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.optim import AdamWConfig, create_optimizer
from fairseq2.optim.lr_scheduler import PolynomialDecayLRConfig, create_lr_scheduler
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.recipes.utils.setup import (
    compile_model,
    setup_root_gang,
    to_data_parallel,
)
from fairseq2.recipes.wav2vec2.common import Wav2Vec2Criterion, Wav2Vec2MetricBag
from fairseq2.recipes.wav2vec2.eval import Wav2Vec2EvalUnit
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import manual_seed

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class Wav2Vec2TrainConfig:
    """Holds the configuration of a wav2vec 2.0 model training task.

    The default values correspond to the base ls960h training setup as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    # Data
    dataset: AssetReference = "librispeech_960h"
    """The name, path or path to the asset card of the speech dataset."""

    train_split: str = "train"
    """The name of the train data split."""

    valid_split: str = "valid"
    """The name of the valid data split."""

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model_family: str = "wav2vec2"
    """The family of the model."""

    model_arch: str | None = "base"
    """The architecture of the wav2vec2 model."""

    model_config: Any = None
    """The configuration of the model."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "stack"
    """The granularity at which to wrap the model."""

    torch_compile: bool = False
    """If ``True``, applies ``torch.compile()`` to the encoder. (experimental)"""

    # Optimizer, LR, and Loss
    optimizer: str = "adamw"
    """The optimizer."""

    optimizer_config: Any = field(
        default_factory=lambda: AdamWConfig(
            lr=5e-04, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01
        )
    )
    """The configuration of the optimizer."""

    lr_scheduler: str = "polynomial-decay"
    """The learning rate scheduler."""

    lr_scheduler_config: Any = field(
        default_factory=lambda: PolynomialDecayLRConfig(num_warmup_steps=32_000)
    )
    """The configuration of the learning rate scheduler."""

    max_gradient_norm: float | None = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""

    # Regime
    max_num_steps: int = 400_000
    """The maximum number of steps to train for."""

    max_num_data_epochs: int | None = None
    """The maximum number of data epochs to train for."""

    validate_every_n_steps: int = 5_000
    """The step interval at which to validate the model."""

    checkpoint_every_n_steps: int = 25_000
    """The step interval at which to checkpoint."""

    keep_best_n_checkpoints: int | None = 1
    """The number of checkpoints to keep based on their validation score. If
    ``None``, none will be deleted."""

    publish_metrics_every_n_steps: int = 200
    """The step interval at which to publish metrics."""

    # Checkpoint
    resume_checkpoint_dir: Path | None = None
    """If not ``None``, adds the specified path to the default asset store."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: tuple[int, int] | None = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, enables the anomaly detection feature of ``torch.autograd``."""


wav2vec2_train_presets = ConfigRegistry[Wav2Vec2TrainConfig]()

wav2vec2_train_preset = wav2vec2_train_presets.decorator


@wav2vec2_train_preset("base_960h")
def _base_960h() -> Wav2Vec2TrainConfig:
    config = Wav2Vec2TrainConfig()

    config.model_config = {"encoder_config": {"first_pass_dropout_p": 0.1}}

    return config


@wav2vec2_train_preset("large_960h")
def _large_960h() -> Wav2Vec2TrainConfig:
    config = Wav2Vec2TrainConfig()

    assert isinstance(config.optimizer_config, AdamWConfig)
    assert isinstance(config.lr_scheduler_config, PolynomialDecayLRConfig)

    config.max_audio_len = 320_000
    config.max_num_elements = 1_200_000
    config.model_arch = "large"
    config.model_config = {"encoder_config": {"first_pass_dropout_p": 0.1}}
    config.optimizer_config.lr = 3e-04
    config.lr_scheduler_config.num_warmup_steps = 20_000
    config.max_num_steps = 250_000
    config.publish_metrics_every_n_steps = 100

    return config


def load_wav2vec2_trainer(
    config: Wav2Vec2TrainConfig, output_dir: Path
) -> Trainer[SequenceBatch]:
    """Load a :class:`Trainer` for wav2vec 2.0 model training."""
    wall_watch = Stopwatch(start=True)

    gang = setup_root_gang(log, monitored=config.monitored_gang)

    checkpoint_manager = FileCheckpointManager(output_dir.joinpath("checkpoints"), gang)

    if config.resume_checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.resume_checkpoint_dir)
        )

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} speech dataset.", dataset_card.name)

        dataset = load_speech_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericSpeechDataset.from_path(dataset_path)

    seed = config.seed

    # Initialize the model
    manual_seed(seed, CPU, gang.device)

    seed += 1

    try:
        model, model_config = create_model(
            config.model_family,
            config.model_arch,
            config.model_config,
            device=META,
            dtype=torch.float32,
        )
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, Wav2Vec2Model):
        raise ValueError(
            f"The model must be of type `{Wav2Vec2Model}`, but is of type `{type(model)}` instead."
        )

    log_model_config(model_config, log)

    checkpoint_manager.save_model_metadata(family=model.family, config=model_config)

    has_checkpoint = checkpoint_manager.has_checkpoint()

    dp_model = to_data_parallel(
        model,
        gang,
        config.data_parallelism,
        log,
        fsdp_broadcast_state=not has_checkpoint,
        fsdp_mixed_precision_dtype=config.dtype,
        fsdp_fp32_reduce=True,
        fsdp_wrap_granularity=config.fsdp_wrap_granularity,
    )

    if config.torch_compile:
        model.encoder = compile_model(model.encoder, log)  # type: ignore[assignment]

    log_model(dp_model, log, rank=gang.rank)

    # Initialize the train criterion.
    criterion = Wav2Vec2Criterion(
        dp_model, config.diversity_loss_weight, config.feature_penalty_weight
    )

    # Initialize the train unit.
    unit = Wav2Vec2TrainUnit(criterion, gang)

    try:
        data_reader = dataset.create_reader(
            config.train_split,
            gang,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            normalize_audio=config.normalize_audio,
            batch_shuffle_window=config.batch_shuffle_window,
            num_accumulate=config.gradient_accumulation,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # Initialize the optimizer.
    try:
        optimizer = create_optimizer(
            config.optimizer, dp_model, config.optimizer_config
        )
    except ValueError as ex:
        raise ValueError(
            "The optimizer cannot be created. See nested exception for details."
        ) from ex

    # Initialize the learning rate scheduler.
    try:
        lr_scheduler = create_lr_scheduler(
            config.lr_scheduler,
            optimizer,
            config.lr_scheduler_config,
            max_num_steps=config.max_num_steps,
        )
    except ValueError as ex:
        raise ValueError(
            "The learning rate scheduler cannot be created. See nested exception for details."
        ) from ex

    # Initialize the validation unit.
    valid_unit = Wav2Vec2EvalUnit(criterion, gang)

    try:
        valid_data_reader = dataset.create_reader(
            config.valid_split,
            gang,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            normalize_audio=config.normalize_audio,
            sync_mode="until_last",
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader for the valid split cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # TODO: Fix once we support static mixed precision on one device.
    amp = gang.size == 1 or config.data_parallelism != "fsdp"

    # Initialize the trainer.
    return Trainer[SequenceBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=gang,
        dtype=config.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=config.fp16_loss_scale,
        max_gradient_norm=config.max_gradient_norm,
        amp=amp,
        max_num_steps=config.max_num_steps,
        max_num_data_epochs=config.max_num_data_epochs,
        score_metric_name="loss",
        lower_better=True,
        valid_units=[valid_unit],
        valid_data_readers=[valid_data_reader],
        validate_after_n_steps=0,
        validate_every_n_steps=config.validate_every_n_steps,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=0,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        keep_best_n_checkpoints=config.keep_best_n_checkpoints,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class Wav2Vec2TrainUnit(AbstractTrainUnit[SequenceBatch]):
    _criterion: Wav2Vec2Criterion
    _metric_bag: Wav2Vec2MetricBag

    def __init__(self, criterion: Wav2Vec2Criterion, gang: Gang) -> None:
        super().__init__(criterion.model)

        self._criterion = criterion

        self._metric_bag = Wav2Vec2MetricBag(gang)

    @override
    def __call__(self, batch: SequenceBatch) -> tuple[Tensor, int]:
        return self._criterion(batch, self._metric_bag)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag

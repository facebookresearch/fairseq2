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
from fairseq2.datasets import LengthBatching
from fairseq2.datasets.asr import GenericAsrDataset, load_asr_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import model_factories
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import load_wav2vec2_model
from fairseq2.models.wav2vec2.asr import (
    Wav2Vec2AsrModel,
    Wav2Vec2AsrOutput,
    wav2vec2_asr_archs,
)
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.optim import AdamWConfig, optimizer_factories
from fairseq2.optim.lr_scheduler import TriStageLRConfig, lr_scheduler_factories
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.recipes.utils.setup import (
    check_model_type,
    compile_model,
    setup_root_gang,
    to_data_parallel,
)
from fairseq2.recipes.wav2vec2.asr.common import Wav2Vec2AsrMetricBag
from fairseq2.recipes.wav2vec2.asr.eval import Wav2Vec2AsrEvalUnit
from fairseq2.typing import META, DataClass, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2AsrTrainConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model training task.

    The default values correspond to the base 10h training setup as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    # Data
    dataset: AssetReference = "librilight_asr_10h"
    """The name, path, or path to the asset card of the ASR dataset."""

    train_split: str = "train"
    """The name of the train data split."""

    valid_split: str = "dev_other"
    """The name of the valid data split."""

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

    tokenizer: AssetReference = "librispeech_asr"
    """The name or path to the asset card of the tokenizer to use."""

    # Model
    pretrained_model: AssetReference = "wav2vec2_base"
    """The name or path to the asset card of the wav2vec 2.0 model to finetune."""

    model_family: str = "wav2vec2_asr"
    """The family of the model."""

    model_arch: Optional[str] = "base_10h"
    """The architecture of the model."""

    model_config: Optional[DataClass] = field(
        default_factory=lambda: wav2vec2_asr_archs.get("base_10h", return_empty=True)
    )
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

    optimizer_config: Optional[DataClass] = field(
        default_factory=lambda: AdamWConfig(lr=5e-05, betas=(0.9, 0.98))
    )
    """The configuration of the optimizer."""

    lr_scheduler: str = "tri-stage"
    """The learning rate scheduler."""

    lr_scheduler_config: Optional[DataClass] = field(
        default_factory=lambda: TriStageLRConfig(
            stage_ratio=(0.1, 0.4, 0.5), start_lr_scale=0.01, final_lr_scale=0.05
        )
    )
    """The configuration of the learning rate scheduler."""

    lr_stage_ratios: tuple[float, float, float] = (0.1, 0.4, 0.5)
    """The ratios of tri-stage learning rate scheduler."""

    start_lr_scale: float = 0.01
    """The scale of the initial warm-up learning rate."""

    final_lr_scale: float = 0.05
    """The scale of the final learning rate."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gradient_accumulation: int = 4
    """The number of steps to accumulate gradients before an optimizer update."""

    # Regime
    max_num_steps: int = 20_000
    """The maximum number of steps to train for."""

    max_num_data_epochs: Optional[int] = None
    """The maximum number of data epochs to train for."""

    freeze_encoder_for_n_steps: int = 10_000
    """The encoder will be frozen for this number of steps."""

    validate_after_n_steps: int = 10_000
    """The number of steps after which to start validating the model."""

    validate_every_n_steps: int = 1000
    """The step interval at which to validate the model."""

    checkpoint_after_n_steps: int = 10_000
    """The number of steps after which to start checkpointing."""

    checkpoint_every_n_steps: int = 1000
    """The step interval at which to checkpoint."""

    keep_best_n_checkpoints: Optional[int] = None
    """The number of checkpoints to keep based on their validation score. If
    ``None``, none will be deleted."""

    publish_metrics_every_n_steps: int = 200
    """The step interval at which to publish metrics."""

    # Checkpoint
    resume_checkpoint_dir: Optional[Path] = None
    """If not ``None``, adds the specified path to the default asset store."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: Optional[tuple[int, int]] = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, enables the anomaly detection feature of ``torch.autograd``."""


wav2vec2_asr_train_presets = ConfigRegistry[Wav2Vec2AsrTrainConfig]()

wav2vec2_asr_train_preset = wav2vec2_asr_train_presets.decorator


@wav2vec2_asr_train_preset("base_10h")
def _base_10h() -> Wav2Vec2AsrTrainConfig:
    return Wav2Vec2AsrTrainConfig()


@wav2vec2_asr_train_preset("base_100h")
def _base_100h() -> Wav2Vec2AsrTrainConfig:
    model_config = wav2vec2_asr_archs.get("base_100h", return_empty=True)

    config = _base_10h()

    assert isinstance(config.optimizer_config, AdamWConfig)

    config.dataset = "librispeech_asr_100h"
    config.model_arch = "base_100h"
    config.model_config = model_config
    config.optimizer_config.lr = 0.00003
    config.max_num_steps = 50_000
    config.freeze_encoder_for_n_steps = 0

    return config


def load_wav2vec2_asr_trainer(
    config: Wav2Vec2AsrTrainConfig, output_dir: Path
) -> Trainer[Seq2SeqBatch]:
    """Load a :class:`Trainer` for wav2vec 2.0 ASR model training."""
    wall_watch = Stopwatch(start=True)

    gang = setup_root_gang(log, monitored=config.monitored_gang)

    checkpoint_manager = FileCheckpointManager(
        output_dir.joinpath("checkpoints"), gang, lower_score_better=True
    )

    if config.resume_checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(
                config.resume_checkpoint_dir, lower_score_better=True
            )
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
        log.info("Loading {} ASR dataset.", dataset_card.name)

        dataset = load_asr_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericAsrDataset.from_path(dataset_path)

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

    if not isinstance(model, Wav2Vec2AsrModel):
        raise ValueError(
            f"The model must be of type `{Wav2Vec2AsrModel}`, but is of type `{type(model)}` instead."
        )

    log_model_config(model_factory.config, log)

    checkpoint_manager.save_model_metadata(
        family=model.family,
        config=model_factory.config,
        tokenizer_name=tokenizer_card.name,
    )

    has_checkpoint = checkpoint_manager.has_checkpoint()

    # If we don't have a checkpoint, load the pretrained model on rank 0 and
    # broadcast it to the gang.
    if not has_checkpoint:
        pretrained_model_card = retrieve_asset_card(config.pretrained_model)

        log.info("Loading pretrained {} model on rank 0.", pretrained_model_card.name)

        if gang.rank == 0:
            pt_model = load_wav2vec2_model(
                pretrained_model_card, device=gang.device, dtype=torch.float32
            )

            share_parameters(pt_model.encoder_frontend, model.encoder_frontend)
            share_parameters(pt_model.encoder, model.encoder)

            if model.masker is not None:
                share_parameters(pt_model.masker, model.masker)

            del pt_model

        gang.barrier()

        log.info("Pretrained model loaded on rank 0.")

        if gang.rank == 0:
            to_device(model, gang.device, seed=seed)

        seed += 1

        gang.barrier()

    # We never train the feature extractor.
    freeze_parameters(model.encoder_frontend.feature_extractor)

    if config.data_parallelism == "fsdp":
        if config.freeze_encoder_for_n_steps != 0:
            raise ValueError("`freeze_encoder_for_n_steps` must be 0 when using FSDP.")

    dp_model = to_data_parallel(
        model,
        gang,
        config.data_parallelism,
        log,
        ddp_find_unused_parameters=config.freeze_encoder_for_n_steps > 0,
        fsdp_skip_init=True,
        fsdp_broadcast_state=not has_checkpoint,
        fsdp_mixed_precision_dtype=config.dtype,
        fsdp_fp32_reduce=True,
        fsdp_wrap_granularity=config.fsdp_wrap_granularity,
    )

    if config.torch_compile:
        model.encoder = compile_model(model.encoder, log)  # type: ignore[assignment]

    log_model(dp_model, log, rank=gang.rank)

    # Initialize the train unit and the optimizer.
    unit = Wav2Vec2AsrTrainUnit(
        dp_model, gang, freeze_encoder_for_n_steps=config.freeze_encoder_for_n_steps
    )

    data_reader = dataset.create_reader(
        config.train_split,
        tokenizer,
        gang,
        batching=LengthBatching(config.max_num_elements),
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        normalize_audio=config.normalize_audio,
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    seed += 1

    try:
        optimizer_factory = optimizer_factories.get(
            config.optimizer, config.optimizer_config
        )

        optimizer = optimizer_factory(dp_model.parameters())
    except ValueError as ex:
        raise ValueError(
            "The optimizer cannot be created. See nested exception for details."
        ) from ex

    try:
        lr_scheduler_factory = lr_scheduler_factories.get(
            config.lr_scheduler, config.lr_scheduler_config
        )

        lr_scheduler = lr_scheduler_factory(optimizer, config.max_num_steps)
    except ValueError as ex:
        raise ValueError(
            "The learning rate scheduler cannot be created. See nested exception for details."
        ) from ex

    # Initialize the validation unit.
    valid_unit = Wav2Vec2AsrEvalUnit(dp_model, gang, tokenizer)

    valid_data_reader = dataset.create_reader(
        config.valid_split,
        tokenizer,
        gang,
        batching=LengthBatching(config.max_num_elements),
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        normalize_audio=config.normalize_audio,
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    seed += 1

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
        score_metric_name="wer",
        lower_better=True,
        valid_units=[valid_unit],
        valid_data_readers=[valid_data_reader],
        validate_after_n_steps=config.validate_after_n_steps,
        validate_every_n_steps=config.validate_every_n_steps,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=config.checkpoint_after_n_steps,
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
class Wav2Vec2AsrTrainUnit(AbstractTrainUnit[Seq2SeqBatch]):
    """Represents a wav2vec 2.0 ASR model training unit."""

    _freeze_encoder_for_n_steps: int
    _metric_bag: Wav2Vec2AsrMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        *,
        freeze_encoder_for_n_steps: int = 0,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 ASR model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        :param freeze_encoder_for_n_steps:
            The encoder will be frozen for this number of steps.
        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2AsrModel)

        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps

        self._metric_bag = Wav2Vec2AsrMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> tuple[Tensor, int]:
        input_batch = SequenceBatch(batch.source_seqs, batch.source_padding_mask)

        output = self._forward(input_batch)

        loss = output.compute_loss(batch.target_seqs, batch.target_padding_mask)

        self._metric_bag.update_ctc_loss(batch, loss.detach())

        self._metric_bag.update_batch_metrics(batch)

        return loss, batch.batch_size

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2AsrOutput:
        return self._model(batch)  # type: ignore[no-any-return]

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

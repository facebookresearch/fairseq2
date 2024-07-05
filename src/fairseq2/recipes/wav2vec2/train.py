# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, final

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.datasets.speech import load_speech_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    create_wav2vec2_model,
    wav2vec2_archs,
)
from fairseq2.nn.utils.module import to_device
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import PolynomialDecayLR
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.log import log_model, log_model_config
from fairseq2.recipes.utils.setup import (
    check_model_type,
    compile_model,
    setup_root_gang,
    to_data_parallel,
    update_model_config,
)
from fairseq2.recipes.wav2vec2.common import Wav2Vec2MetricBag
from fairseq2.recipes.wav2vec2.eval import Wav2Vec2EvalUnit
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2TrainConfig:
    """Holds the training configuration of a wav2vec 2.0 model.

    The default values correspond to the base ls960h training setup as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    # Data
    dataset: Union[str, Path] = "librispeech_960h"
    """The name or path to the asset card of the dataset to train with."""

    train_split: str = "train"
    """The name of the dataset split to train with."""

    valid_split: str = "valid"
    """The name of the dataset split to validate with."""

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    example_shuffle_window: int = 1
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model_arch: Optional[str] = "base"
    """The architecture of the wav2vec2 model."""

    model_config: Optional[Dict[str, Any]] = None
    """The model configuration overrides."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "stack"
    """The granularity at which to wrap the model."""

    torch_compile: bool = False
    """If ``True``, applies ``torch.compile()`` to the encoder. (experimental)"""

    # Optimizer, LR, and Loss
    lr: float = 5e-04
    """The initial (post-warm-up) learning rate."""

    warmup_steps: int = 32_000
    """The number of warmup steps for polynomial LR scheduler."""

    betas: Tuple[float, float] = (0.9, 0.98)
    """The coefficients of AdamW."""

    eps: float = 1e-06
    """Adam's epsilon parameter."""

    weight_decay: float = 0.01
    """The weight decay parameter used for regularization."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: Tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    # Regime
    max_num_steps: int = 400_000
    """The maximum number of steps to train for."""

    max_num_data_epochs: Optional[int] = None
    """The maximum number of data epochs to train for."""

    validate_every_n_steps: int = 1000
    """The step interval at which to validate the model."""

    checkpoint_after_n_steps: int = 0
    """The number of steps after which to start checkpointing."""

    checkpoint_every_n_steps: int = 25_000
    """The step interval at which to checkpoint."""

    publish_metrics_every_n_steps: int = 1000
    """The step interval at which to publish metrics."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: Optional[Tuple[int, int]] = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, enables the anomaly detection feature of ``torch.autograd``."""


wav2vec2_train_presets = ConfigRegistry[Wav2Vec2TrainConfig]()

wav2vec2_train_preset = wav2vec2_train_presets.decorator


@wav2vec2_train_preset("base_960h")
def _base_960h() -> Wav2Vec2TrainConfig:
    return Wav2Vec2TrainConfig()


def load_wav2vec2_trainer(
    config: Wav2Vec2TrainConfig, output_dir: Path
) -> Trainer[Tensor]:
    """Load a :class:`Trainer` for wav2vec 2.0 model training."""
    wall_watch = Stopwatch(start=True)

    gang = setup_root_gang(log, monitored=config.monitored_gang)

    checkpoint_manager = FileCheckpointManager(output_dir.joinpath("checkpoints"), gang)

    # Load the dataset.
    dataset_card = retrieve_asset_card(config.dataset)

    log.info("Loading {} speech dataset.", dataset_card.name)

    dataset = load_speech_dataset(config.dataset)

    log.info("Dataset loaded.")

    # Initialize the model configuration.
    if config.model_arch is None:
        model_config = Wav2Vec2Config()
    else:
        model_config = wav2vec2_archs.get(config.model_arch)

    if config.model_config is not None:
        update_model_config(model_config, config.model_config)

    log_model_config(model_config, log)

    # Initialize the model.
    model = create_wav2vec2_model(model_config, device=META, dtype=torch.float32)

    checkpoint_manager.save_model_metadata(family=model.family, config=model_config)

    has_checkpoint = checkpoint_manager.has_checkpoint()

    # If we don't have a checkpoint, load the model on rank 0 and
    # broadcast it to the gang.
    if not has_checkpoint:
        if gang.rank == 0:
            to_device(model, gang.device, seed=config.seed)

        gang.barrier()

    dp_model = to_data_parallel(
        model,
        gang,
        config.data_parallelism,
        log,
        ddp_find_unused_parameters=False,
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
    unit = Wav2Vec2TrainUnit(dp_model, gang)

    data_reader = dataset.create_reader(
        config.train_split,
        gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    optimizer = AdamW(
        dp_model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = PolynomialDecayLR(
        optimizer,
        config.max_num_steps,
        config.warmup_steps,
    )

    # Initialize the validation unit.
    valid_unit = Wav2Vec2EvalUnit(dp_model, gang)

    valid_data_reader = dataset.create_reader(
        config.valid_split,
        gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    # Initialize the trainer.
    return Trainer[Tensor](
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
        valid_units=[valid_unit],
        valid_data_readers=[valid_data_reader],
        validate_after_n_steps=0,
        validate_every_n_steps=config.validate_every_n_steps,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=config.checkpoint_after_n_steps,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        tb_dir=output_dir.joinpath("tb"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class Wav2Vec2TrainUnit(AbstractTrainUnit[Tensor]):
    """Represents the training unit of a wav2vec 2.0 model."""

    _metric_bag: Wav2Vec2MetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.

        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2Model)

        self._metric_bag = Wav2Vec2MetricBag(gang)

    @override
    def __call__(self, seqs: Tensor) -> Tuple[Tensor, int]:
        batch = SequenceBatch(seqs, None)

        output = self._model(batch)

        loss = output.compute_loss()

        self._metric_bag.update_loss(batch, loss.detach())

        self._metric_bag.update_batch_metrics(batch)

        return loss.total, batch.batch_size

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag

    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_source_elements"

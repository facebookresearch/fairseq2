# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from torch.nn import Module

from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.asr import load_asr_dataset
from fairseq2.logging import get_log_writer
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.wav2vec2 import load_wav2vec2_model
from fairseq2.models.wav2vec2.asr import (
    Wav2Vec2AsrConfig,
    create_wav2vec2_asr_model,
    wav2vec2_asr_archs,
)
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import TriStageLR
from fairseq2.recipes.trainer import StandardTrainer
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import setup_gangs
from fairseq2.recipes.wav2vec2.asr.criterion import Wav2Vec2AsrCriterion
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


@dataclass
class Wav2Vec2AsrTrainConfig:
    """Holds the configuration of a wav2vec 2.0 ASR training recipe.

    The default values correspond to the base 10h training as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    # Data
    dataset_name: str = "librilight_asr_10h"
    """The dataset to train with."""

    tokenizer_name: str = "librispeech_asr"
    """The tokenizer to use."""

    train_split: str = "train"
    """The name of the dataset split to train with."""

    valid_split: str = "dev_other"
    """The name of the dataset split to validate with."""

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    shuffle_window_size: int = 1000
    """The size of the sliding data shuffle window."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    pretrained_model_name: str = "wav2vec2_base"
    """The name of the wav2vec 2.0 model to finetune."""

    model_config: Wav2Vec2AsrConfig = field(
        default_factory=lambda: wav2vec2_asr_archs.get("base_10h")
    )
    """The configuration of the ASR model."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "ddp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "stack"
    """The granularity at which to wrap the ASR model."""

    torch_compile: bool = False
    """If ``True``, applies ``torch.compile()`` to the encoder. (experimental)"""

    # Optimizer, LR, and Loss
    lr: float = 5e-05
    """The initial (post-warm-up) learning rate."""

    betas: Tuple[float, float] = (0.9, 0.98)
    """The coefficients of AdamW."""

    lr_stage_ratios: Tuple[float, float, float] = (0.1, 0.4, 0.5)
    """The ratios of tri-stage learning rate scheduler."""

    start_lr_scale: float = 0.01
    """The scale of the initial warm-up learning rate."""

    final_lr_scale: float = 0.05
    """The scale of the final learning rate."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: Tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    gradient_accumulation: int = 4
    """The number of steps to accumulate gradients before an optimizer update."""

    # Regime
    max_num_steps: int = 20_000
    """The maximum number of training steps."""

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

    publish_metrics_every_n_steps: int = 200
    """The step interval at which to publish metrics."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: bool = False
    """If ``True``, runs the PyTorch profiler early in the training."""

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
    config = _base_10h()

    config.dataset_name = "librispeech_asr_100h"

    config.model_config = wav2vec2_asr_archs.get("base_100h")

    config.lr = 0.00003
    config.max_num_steps = 50_000
    config.freeze_encoder_for_n_steps = 0

    return config


def load_wav2vec2_asr_trainer(
    config: Wav2Vec2AsrTrainConfig, output_dir: Path
) -> StandardTrainer[Seq2SeqBatch]:
    """Load a wav2vec 2.0 ASR tainer."""
    wall_watch = Stopwatch(start=True)

    gang, _ = setup_gangs(log, monitored=config.monitored_gang)

    log.info("Loading {} tokenizer.", config.tokenizer_name)

    tokenizer = load_text_tokenizer(config.tokenizer_name)

    if config.model_config.final_dim != tokenizer.vocab_info.size:
        raise ValueError(
            f"`config.model_config.final_dim` must match the size of the vocabulary of the tokenizer ({tokenizer.vocab_info.size}), but is {config.model_config.final_dim} instead."
        )

    log.info("Tokenizer loaded.")

    log.info("Loading {} dataset.", config.dataset_name)

    dataset = load_asr_dataset(config.dataset_name)

    train_data_reader = dataset.create_reader(
        split=config.train_split,
        tokenizer=tokenizer,
        gang=gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        shuffle_window_size=config.shuffle_window_size,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    valid_data_reader = dataset.create_reader(
        split=config.valid_split,
        tokenizer=tokenizer,
        gang=gang,
        dtype=config.dtype,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        max_num_elements=config.max_num_elements,
        normalize_audio=config.normalize_audio,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    data_readers = {"train": train_data_reader, "valid": valid_data_reader}

    log.info("Dataset loaded.")

    # Set up the checkpoint manager.
    if gang.size > 1 and config.data_parallelism == "ddp":
        replicated_keys = ["_model", "_optimizer"]
    else:
        replicated_keys = []

    checkpoint_manager = FileCheckpointManager(
        output_dir.joinpath("checkpoints"),
        gang,
        model_key="_model",
        replicated_keys=replicated_keys,
    )

    rng_bag = RngBag.from_device_defaults(CPU, gang.device)

    # Set the seed for model initialization.
    rng_bag.manual_seed(config.seed)

    model = create_wav2vec2_asr_model(
        config.model_config, device=META, dtype=torch.float32
    )

    has_checkpoint = checkpoint_manager.has_checkpoint()

    # If we don't have a checkpoint, load the pretrained model on rank 0 and
    # broadcast it to the gang.
    if not has_checkpoint:
        log.info("Loading pretrained {} model on rank 0.", config.pretrained_model_name)

        if gang.rank == 0:
            pt_model = load_wav2vec2_model(
                config.pretrained_model_name, device=gang.device, dtype=torch.float32
            )

            share_parameters(pt_model.encoder_frontend, model.encoder_frontend)
            share_parameters(pt_model.encoder, model.encoder)

            if model.masker is not None:
                share_parameters(pt_model.masker, model.masker)

            del pt_model

        gang.barrier()

        log.info("Pretrained model loaded on rank 0 and parameters shared with the ASR model.")  # fmt: skip

        log.info("Initialize the output linear layer on rank 0.")

        if gang.rank == 0:
            to_device(model, gang.device)

        gang.barrier()

        log.info("Output linear layer initialized.")

    checkpoint_manager.set_model_metadata(
        family=model.family, config=config.model_config
    )

    # We never train the feature extractor.
    freeze_parameters(model.encoder_frontend.feature_extractor)

    dp_model: Module

    # Set up data parallelism.
    if gang.size == 1:
        to_device(model, gang.device)

        dp_model = model
    else:
        log.info("Wrapping the model with {} and broadcasting to all ranks from rank 0.", config.data_parallelism.upper())  # fmt: skip

        if config.data_parallelism == "ddp":
            to_device(model, gang.device)

            find_unused_params = config.freeze_encoder_for_n_steps > 0

            dp_model = to_ddp(model, gang, find_unused_parameters=find_unused_params)
        elif config.data_parallelism == "fsdp":
            if config.freeze_encoder_for_n_steps != 0:
                raise ValueError(
                    "`config.freeze_encoder_for_n_steps` must be 0 when using FSDP."
                )

            wrap_policy, ignored_modules = get_fsdp_wrap_policy(
                model, wrap_granularity=config.fsdp_wrap_granularity
            )

            if config.dtype == torch.float32:
                mixed_precision_dtype = None
            else:
                mixed_precision_dtype = config.dtype

            dp_model = to_fsdp(
                model,
                gang,
                wrap_policy,
                ignored_modules=ignored_modules,
                skip_init=True,
                broadcast_state=not has_checkpoint,
                mixed_precision_dtype=mixed_precision_dtype,
                fp32_reduce=True,
            )
        else:
            raise ValueError(
                f"`config.data_parallelism` must be 'ddp' or 'fsdp', but is '{config.data_parallelism}' instead."
            )

        log.info("Model wrapped and broadcasted to all ranks.")

    if config.torch_compile:
        log.info("Compiling the encoder.")

        model.encoder = torch.compile(  # type: ignore[assignment]
            model.encoder, dynamic=True, options={"shape_padding": True}
        )

        gang.barrier()

        log.info("Encoder compiled.")

    log_model(dp_model, log)

    # Initialize the criterion and the optimizer.
    criterion = Wav2Vec2AsrCriterion(
        dp_model,
        gang,
        tokenizer,
        freeze_encoder_for_n_steps=config.freeze_encoder_for_n_steps,
    )

    optimizer = AdamW(dp_model.parameters(), lr=config.lr, betas=config.betas)

    lr_scheduler = TriStageLR(
        optimizer,
        config.max_num_steps,
        config.lr_stage_ratios,
        start_lr_scale=config.start_lr_scale,
        final_lr_scale=config.final_lr_scale,
    )

    # Initialize the profiler.
    tb_dir = output_dir.joinpath("tb")

    profiler = Profiler(
        skip_first=15, active=3, log_dir=tb_dir, gang=gang, enabled=config.profile
    )

    # Set the seed for training.
    rng_bag.manual_seed(config.seed + gang.rank)

    # Initialize the trainer.
    return StandardTrainer[Seq2SeqBatch](
        criterion=criterion,
        gang=gang,
        dtype=config.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=config.fp16_loss_scale,
        max_gradient_norm=config.max_gradient_norm,
        data_readers=data_readers,
        max_num_steps=config.max_num_steps,
        validate_after_n_steps=config.validate_after_n_steps,
        validate_every_n_steps=config.validate_every_n_steps,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=config.checkpoint_after_n_steps,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        tb_dir=tb_dir,
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profiler=profiler,
        anomaly_detection=config.anomaly_detection,
        rng_bag=rng_bag,
        wall_watch=wall_watch,
    )

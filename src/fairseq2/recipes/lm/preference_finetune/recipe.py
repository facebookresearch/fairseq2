# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
import torch.distributed
from torch.nn import Module

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider, FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets import Batching, LengthBatching, StaticBatching
from fairseq2.datasets.preference import (
    GenericPreferenceOptimizationDataset,
    PreferenceOptimizationBatch,
    load_preference_optimization_dataset,
)
from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import AdamWConfig, create_optimizer
from fairseq2.optim.lr_scheduler import CosineAnnealingLRConfig, create_lr_scheduler
from fairseq2.recipes.trainer import Trainer, TrainUnit
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import compile_model, setup_gangs, to_data_parallel
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import manual_seed

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class PreferenceOptimizationConfig:
    """Holds the configuration of a language model preference-finetuning task."""

    # Data
    dataset: AssetReference = "openeft"  # TODO: change!
    """The name, path, or path to the asset card of the preference optimization dataset."""

    max_seq_len: int = 8192
    """The maximum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Longer sequences will be dropped."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of total `src`, `tgt_chosen`, and `tgt_rejected` tokens per batch."""

    batch_size: int | None = None
    """If not ``None``, ignores `max_num_tokens` and each batch will have `batch_size` examples."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    mask_source_tokens: bool = True
    """If ``False``, calculates loss on the `src` tokens as well as the `tgt` tokens."""

    # Model
    model: AssetReference = "llama3_8b_instruct"
    """The name or path to the asset card of the language model to finetune."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "fsdp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "layer"
    """The granularity at which to wrap the model."""

    fsdp_reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    tensor_parallel_size: int = 1
    """The size of tensor parallelism."""

    activation_checkpointing: bool = True
    """If ``True``, uses layer-wise activation checkpointing."""

    torch_compile: bool = False
    """If ``True``, applies ``torch.compile()`` to the decoder. (experimental)"""

    # Criterion
    criterion: str = "dpo"
    """The preference optimization criterion."""

    criterion_config: Any = None
    """The configuration of the preference optimization criterion."""

    # Optimizer, LR, and Loss
    optimizer: str = "adamw"
    """The optimizer."""

    optimizer_config: Any = field(
        default_factory=lambda: AdamWConfig(
            lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1
        )
    )
    """The configuration of the optimizer."""

    lr_scheduler: str = "cosine-annealing"
    """The learning rate scheduler."""

    lr_scheduler_config: Any = field(
        default_factory=lambda: CosineAnnealingLRConfig(final_lr=5.5e-06 * 0.2)
    )
    """The configuration of the learning rate scheduler."""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    max_gradient_norm: float | None = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    # Regime
    max_num_steps: int = 5000
    """The maximum number of steps to train for."""

    max_num_data_epochs: int | None = None
    """The maximum number of data epochs to train for."""

    checkpoint_every_n_steps: int = 1000
    """The step interval at which to checkpoint."""

    checkpoint_every_n_data_epochs: int | None = None
    """The data epoch interval at which to checkpoint."""

    keep_last_n_checkpoints: int | None = 1
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    keep_last_n_models: int | None = None
    """The number of checkpoint models to keep."""

    publish_metrics_every_n_steps: int = 10
    """The step interval at which to publish training metrics."""

    publish_metrics_every_n_data_epochs: int | None = None
    """The data epoch interval at which to publish training metrics."""

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
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""


preference_finetune_presets = ConfigRegistry[PreferenceOptimizationConfig]()

preference_finetune_preset = preference_finetune_presets.decorator


# @preference_finetune_preset("simpo")
# def _simpo() -> PreferenceOptimizationConfig:
#    cfg = PreferenceOptimizationConfig()
#    cfg.max_num_tokens = 1200
#    cfg.max_seq_len = 600
#    cfg.model = "llama3_8b"
#    cfg.simpo = SimpoFinetuneConfig()
#    return cfg


@preference_finetune_preset("llama3_8b_instruct")
def _llama3_8b_instruct() -> PreferenceOptimizationConfig:
    config = PreferenceOptimizationConfig()

    config.max_seq_len = 1000
    config.max_num_tokens = 1000
    config.max_gradient_norm = 1.0

    return config


# batch size and min lengths are tuned for OA2 in this preset!
@preference_finetune_preset("llama3_70b_instruct_openassistant2")
def _llama3_70b_instruct_openassistant2() -> PreferenceOptimizationConfig:
    config = PreferenceOptimizationConfig()

    # 70B DPO training might catch OOM, tune the effective batch size if needed.
    config.max_seq_len = 200
    config.max_num_tokens = 200
    config.model = "llama3_70b_instruct"
    config.tensor_parallel_size = 8
    config.max_gradient_norm = 1.0
    config.gradient_accumulation = 8  # to address small batch size

    return config


def load_preference_finetuner(
    config: PreferenceOptimizationConfig, output_dir: Path
) -> Trainer[PreferenceOptimizationBatch]:
    """Load a :class:`Trainer` for language model preference optimization-finetuning."""
    wall_watch = Stopwatch(start=True)

    root_gang, gangs = setup_gangs(
        log, tp_size=config.tensor_parallel_size, monitored=config.monitored_gang
    )

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    checkpoint_manager = FileCheckpointManager(
        output_dir.joinpath("checkpoints"), root_gang, dp_gang=dp_gang, tp_gang=tp_gang
    )

    if config.resume_checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.resume_checkpoint_dir)
        )

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} preference optimization dataset.", dataset_card.name)

        dataset = load_preference_optimization_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericPreferenceOptimizationDataset.from_path(dataset_path)

    seed = config.seed

    # Load the model
    manual_seed(seed, CPU, root_gang.device)

    seed += 1

    init_device = META

    has_checkpoint = checkpoint_manager.has_checkpoint()

    if has_checkpoint:
        try:
            model = load_model(
                model_card, gangs=gangs, device=init_device, dtype=torch.float32
            )
        except ValueError as ex:
            raise ValueError(
                "The model cannot be initialized. See nested exception for details."
            ) from ex
    # If we don't have a checkpoint, load the pretrained model on rank 0 and
    # broadcast it to the gang.
    else:
        log.info("Loading {} model on data parallel rank 0 (per shard).", model_card.name)  # fmt: skip

        if dp_gang.rank == 0:
            init_device = root_gang.device

        model = load_model(
            model_card, gangs=gangs, device=init_device, dtype=torch.float32
        )

        root_gang.barrier()

        log.info("Model loaded on data parallel rank 0.")

    if not isinstance(model, DecoderModel):
        raise ValueError(
            f"The model must be of type `{DecoderModel}`, but is of type `{type(model)}` instead."
        )

    checkpoint_manager.save_model_metadata(base_asset=model_card.name)

    dp_model = to_data_parallel(
        model,
        dp_gang,
        config.data_parallelism,
        log,
        fsdp_skip_init=True,
        fsdp_broadcast_state=not has_checkpoint,
        fsdp_reshard_after_forward=config.fsdp_reshard_after_forward,
        fsdp_mixed_precision_dtype=config.dtype,
        fsdp_fp32_reduce=True,
        fsdp_wrap_granularity=config.fsdp_wrap_granularity,
    )

    if config.activation_checkpointing:
        use_layerwise_activation_checkpointing(dp_model)

    if config.torch_compile:
        model.decoder = compile_model(model.decoder, log)

    # TODO(balioglu): investigate!
    # The memory efficient SDPA implementation in PyTorch is not stable when
    # used with padded inputs.
    enable_memory_efficient_torch_sdpa(dp_model, False)

    log_model(dp_model, log, rank=root_gang.rank)

    # Initialize the train unit.
    try:
        unit_factory = preference_unit_factories.get(
            config.criterion, config.criterion_config
        )

        unit = unit_factory(dp_model, root_gang, gangs)
    except ValueError as ex:
        raise ValueError(
            "The criterion cannot be initialized. See nested exception for details."
        ) from ex

    # Initialize the data reader.
    batching: Batching

    if config.batch_size is not None:
        batching = StaticBatching(config.batch_size)
    else:
        batching = LengthBatching(config.max_num_tokens)

    try:
        data_reader = dataset.create_reader(
            tokenizer,
            dp_gang,
            max_seq_len=config.max_seq_len,
            batching=batching,
            max_num_tokens=config.max_num_tokens,
            example_shuffle_window=config.example_shuffle_window,
            batch_shuffle_window=config.batch_shuffle_window,
            num_accumulate=config.gradient_accumulation,
            num_prefetch=config.num_prefetch,
            mask_source_tokens=config.mask_source_tokens,
            seed=config.seed,
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

    # Initialize the trainer.
    return Trainer[PreferenceOptimizationBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        dtype=config.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=config.fp16_loss_scale,
        max_gradient_norm=config.max_gradient_norm,
        max_num_steps=config.max_num_steps,
        max_num_data_epochs=config.max_num_data_epochs,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        checkpoint_every_n_data_epochs=config.checkpoint_every_n_data_epochs,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        keep_last_n_models=config.keep_last_n_models,
        tb_dir=output_dir.joinpath("tb"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        publish_metrics_every_n_data_epochs=config.publish_metrics_every_n_data_epochs,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=config.seed,
        wall_watch=wall_watch,
    )


preference_unit_factories = ConfigBoundFactoryRegistry[
    [Module, Gang, Mapping[str, Gang]], TrainUnit[PreferenceOptimizationBatch]
]()

preference_unit_factory = preference_unit_factories.decorator

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, cast, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module

from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.instruction import load_instruction_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import MetricBag
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import CosineAnnealingLR
from fairseq2.recipes.criterion import AbstractCriterion
from fairseq2.recipes.metrics import SequenceModelMetricBag
from fairseq2.recipes.trainer import StandardTrainer
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import compile_model, setup_gangs, to_data_parallel
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class InstructionFinetuneConfig:
    """Holds the configuration of an instruction-finetuning recipe."""

    # Data
    dataset: Union[str, Path] = "openeft"  # TODO: change!
    """The name or path to the asset card of the dataset to train with."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    tokenizer: Union[str, Path] = "llama3_instruct"
    """The name or path to the asset card of the tokenizer to use."""

    # Model
    model: Union[str, Path] = "llama3_8b_instruct"
    """The name or path to the asset card of the model to finetune."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "fsdp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "layer"
    """The granularity at which to wrap the model."""

    fsdp_reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    tensor_parallel_size: int = 1
    """The size of Megatron-style tensor parallelism."""

    activation_checkpointing: bool = True
    """If ``True``, uses layer-wise activation checkpointing."""

    torch_compile: bool = False
    """If ``True``, applies ``torch.compile()`` to the decoder. (experimental)"""

    # Optimizer, LR, and Loss
    lr: float = 5.5e-06
    """The initial (post-warm-up) learning rate."""

    betas: Tuple[float, float] = (0.9, 0.95)
    """The coefficients of AdamW."""

    final_lr_ratio: float = 0.2
    """The ratio of the final learning rate to :attr:`lr`."""

    weight_decay: float = 0.1
    """The weight decay coefficient of AdamW."""

    num_lr_warmup_steps: int = 0
    """The number of learning rate warm-up steps."""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: Tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    # Regime
    max_num_steps: int = 5000
    """The maximum number of steps to train for."""

    max_num_data_epochs: Optional[int] = None
    """The maximum number of data epochs to train for."""

    checkpoint_every_n_steps: int = 1000
    """The step interval at which to checkpoint."""

    keep_last_n_checkpoints: Optional[int] = 1
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    keep_last_n_models: Optional[int] = None
    """The number of checkpoint models to keep."""

    publish_metrics_every_n_steps: int = 10
    """The step interval at which to publish training metrics."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: Optional[Tuple[int, int]] = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""


instruction_finetune_presets = ConfigRegistry[InstructionFinetuneConfig]()

instruction_finetune_preset = instruction_finetune_presets.decorator


@instruction_finetune_preset("llama3_8b_instruct")
def _llama3_8b_instruct() -> InstructionFinetuneConfig:
    return InstructionFinetuneConfig()


@instruction_finetune_preset("llama3_70b_instruct")
def _llama3_70b_instruct() -> InstructionFinetuneConfig:
    config = _llama3_8b_instruct()

    config.model = "llama3_70b_instruct"
    config.tensor_parallel_size = 8

    return config


@instruction_finetune_preset("llama2_7b_chat")
def _llama2_7b_chat() -> InstructionFinetuneConfig:
    config = _llama3_8b_instruct()

    config.tokenizer = "llama2"
    config.max_seq_len = 4096
    config.max_num_tokens = 4096 * 2
    config.model = "llama2_7b_chat"

    return config


@instruction_finetune_preset("llama2_70b_chat")
def _llama2_70b_chat() -> InstructionFinetuneConfig:
    config = _llama2_7b_chat()

    config.model = "llama2_70b_chat"
    config.tensor_parallel_size = 8

    return config


def load_instruction_finetuner(
    config: InstructionFinetuneConfig, output_dir: Path
) -> StandardTrainer[SequenceBatch]:
    """Load a :class:`Trainer` for instruction finetuning."""
    wall_watch = Stopwatch(start=True)

    root_gang, gangs = setup_gangs(
        log, tp_size=config.tensor_parallel_size, monitored=config.monitored_gang
    )

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    # Load the tokenizer.
    tokenizer = load_text_tokenizer(config.tokenizer)

    # Load the data reader.
    dataset = load_instruction_dataset(config.dataset)

    data_reader = dataset.create_reader(
        split="train",
        tokenizer=tokenizer,
        gang=dp_gang,
        max_seq_len=config.max_seq_len,
        max_num_tokens=config.max_num_tokens,
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    data_readers = {"train": data_reader}

    # Initialize the model.
    model_card = retrieve_asset_card(config.model)

    init_device = META

    # Set up the checkpoint manager.
    checkpoint_manager = FileCheckpointManager(
        output_dir.joinpath("checkpoints"), root_gang, dp_gang=dp_gang, tp_gang=tp_gang
    )

    has_checkpoint = checkpoint_manager.has_checkpoint()

    if has_checkpoint:
        model = load_model(
            model_card, gangs=gangs, device=init_device, dtype=torch.float32
        )
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
        raise ValueError("`config.model` must specify a decoder model.")

    if model.vocab_info != tokenizer.vocab_info:
        raise ValueError(
            "`vocab_info` of the model and `vocab_info` of the tokenizer do not match."
        )

    checkpoint_manager.save_model_metadata(
        base_asset=model_card.name, family=model.family
    )

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

    # Initialize the criterion and the optimizer.
    criterion = InstructionFinetuneCriterion(dp_model, dp_gang)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        cycle_len=config.max_num_steps - config.num_lr_warmup_steps,
        num_warmup_steps=config.num_lr_warmup_steps,
        final_lr=config.lr * config.final_lr_ratio,
    )

    # Set up the finetuner.
    return StandardTrainer[SequenceBatch](
        criterion=criterion,
        gang=root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        dtype=config.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=config.fp16_loss_scale,
        max_gradient_norm=config.max_gradient_norm,
        data_readers=data_readers,
        max_num_steps=config.max_num_steps,
        max_num_data_epochs=config.max_num_data_epochs,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        keep_last_n_models=config.keep_last_n_models,
        tb_dir=output_dir.joinpath("tb"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class InstructionFinetuneCriterion(AbstractCriterion[SequenceBatch]):
    """Computes cross entropy loss of a Language Model."""

    _train_metric_bag: SequenceModelMetricBag
    _valid_metric_bag: MetricBag

    def __init__(self, model: Module, gang: Gang) -> None:
        super().__init__(model)

        self._train_metric_bag = SequenceModelMetricBag(gang)
        self._valid_metric_bag = MetricBag(gang)

    @override
    def compute_loss(self, batch: SequenceBatch) -> Tuple[Tensor, int]:
        batch, target_batch = as_auto_regressive_input(batch)

        output = cast(SequenceModelOutput, self._model(batch))

        loss = output.compute_loss(
            target_batch.seqs, loss_mask=target_batch.target_mask
        )

        if self._model.training:
            self._train_metric_bag.update_loss_metrics(target_batch, loss)

        return loss, target_batch.num_target_elements()

    @final
    @property
    @override
    def train_metric_bag(self) -> SequenceModelMetricBag:
        return self._train_metric_bag

    @final
    @property
    @override
    def valid_metric_bag(self) -> MetricBag:
        return self._valid_metric_bag

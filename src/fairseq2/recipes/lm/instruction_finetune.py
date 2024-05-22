# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, cast, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module

from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.instruction import load_instruction_dataset
from fairseq2.gang import Gang, setup_default_gang, setup_parallel_gangs
from fairseq2.logging import get_log_writer
from fairseq2.metrics import MetricBag
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.nn.utils.module import to_device
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import CosineAnnealingLR
from fairseq2.recipes.criterion import AbstractCriterion
from fairseq2.recipes.metrics import SequenceModelMetricBag
from fairseq2.recipes.trainer import StandardTrainer, Trainer
from fairseq2.recipes.utils.log import log_environment_info, log_model
from fairseq2.typing import CPU, META, DataType, override
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


@dataclass
class InstructionFinetuneConfig:
    """Holds the configuration of an instruction-finetuning job."""

    # Data
    dataset_name: str = "foo"  # TODO: fix!
    """The dataset to train with."""

    tokenizer_name: str = "llama2_7b_chat"
    """The tokenizer to use."""

    max_seq_len: int = 4096
    """The maximum sequence length."""

    max_num_tokens: int = 8192
    """The maximum number of tokens per batch."""

    shuffle_window_size: int = 10_000
    """The size of the sliding data shuffle window."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model_name: str = "llama2_7b_chat"
    """The name of the model to finetune."""

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

    num_lr_warmup_steps: int = 1  # TODO: 2000
    """The number of learning rate warm-up steps."""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    max_gradient_norm: Optional[float] = None
    """The maximum gradient norm. If ``None``, no clipping will be applied."""

    fp16_loss_scale: Tuple[float, float] = (128.0, 0.0001)
    """The initial and minimum loss scale for fp16 training."""

    # Regime
    max_num_steps: int = 5000
    """The maximum number of training steps."""

    checkpoint_every_n_steps: int = 1000
    """The step interval at which to checkpoint."""

    keep_last_n_checkpoints: Optional[int] = 1
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    publish_metrics_every_n_steps: int = 10
    """The step interval at which to publish training metrics."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""

    profile: bool = False
    """If ``True``, runs the PyTorch profiler early in the training."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""


instruction_finetune_presets = ConfigRegistry[InstructionFinetuneConfig]()

instruction_finetune_preset = instruction_finetune_presets.decorator


@instruction_finetune_preset("llama2_7b_chat")
def _llama2_7b_chat() -> InstructionFinetuneConfig:
    return InstructionFinetuneConfig()


def load_instruction_finetuner(
    config: InstructionFinetuneConfig, output_dir: Path
) -> Trainer:
    wall_watch = Stopwatch(start=True)

    # In case we train on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    # Set up the root gang.
    root_gang = setup_default_gang(monitored=config.monitored_gang)

    # Set up the gangs for data and tensor parallelism.
    try:
        gangs = setup_parallel_gangs(root_gang, tp_size=config.tensor_parallel_size)
    except ValueError as ex:
        raise RuntimeError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `config.tensor_parallel_size` ({config.tensor_parallel_size})."
        ) from ex

    device = root_gang.device

    log_environment_info(log, device)

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    # Load the tokenizer.
    tokenizer = load_text_tokenizer(config.tokenizer_name)

    # Load the dataset.
    dataset = load_instruction_dataset(config.dataset_name)

    data_reader = dataset.create_reader(
        split="train",
        tokenizer=tokenizer,
        gang=dp_gang,
        max_seq_len=config.max_seq_len,
        max_num_tokens=config.max_num_tokens,
        shuffle_window_size=config.shuffle_window_size,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

    data_readers = {"train": data_reader}

    # Set up the checkpoint manager.
    if dp_gang.size > 1 and config.data_parallelism == "ddp":
        replicated_keys = ["_model", "_optimizer"]
    else:
        replicated_keys = []

    checkpoint_manager = FileCheckpointManager(
        output_dir.joinpath("checkpoints"),
        root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        model_key="_model",
        replicated_keys=replicated_keys,
    )

    rng_bag = RngBag.from_device_defaults(CPU, device)

    # Set the seed for model initialization.
    rng_bag.manual_seed(config.seed)

    # Initialize the model.
    log.info("Initializing the model.")

    has_checkpoint = checkpoint_manager.has_checkpoint()

    # If we don't have a checkpoint, load the pretrained model on rank 0 and
    # broadcast it to the gang.
    if not has_checkpoint and dp_gang.rank == 0:
        init_device = device
    else:
        init_device = META

    model = load_model(
        config.model_name, gangs=gangs, device=init_device, dtype=torch.float32
    )

    root_gang.barrier()

    if not isinstance(model, DecoderModel):
        raise ValueError("`config.model_name` must specify a decoder model.")

    if model.vocab_info != tokenizer.vocab_info:
        raise ValueError(
            "`vocab_info` of the model and `vocab_info` of the tokenizer do not match."
        )

    checkpoint_manager.set_model_metadata(family=model.family, base=config.model_name)

    dp_model: Module

    # Set up data parallelism.
    if dp_gang.size == 1:
        to_device(model, device)

        dp_model = model
    elif config.data_parallelism == "ddp":
        to_device(model, device)

        dp_model = to_ddp(model, dp_gang)
    elif config.data_parallelism == "fsdp":
        wrap_policy, ignored_modules = get_fsdp_wrap_policy(
            model, wrap_granularity=config.fsdp_wrap_granularity
        )

        if config.dtype == torch.float32:
            mixed_precision_dtype = None
        else:
            mixed_precision_dtype = config.dtype

        dp_model = to_fsdp(
            model,
            dp_gang,
            wrap_policy,
            ignored_modules=ignored_modules,
            skip_init=True,
            broadcast_state=not has_checkpoint,
            reshard_after_forward=config.fsdp_reshard_after_forward,
            mixed_precision_dtype=mixed_precision_dtype,
            fp32_reduce=True,
        )
    else:
        raise ValueError(
            f"`config.data_parallelism` must be 'ddp' or 'fsdp', but is '{config.data_parallelism}' instead."
        )

    if config.activation_checkpointing:
        use_layerwise_activation_checkpointing(dp_model)

    if config.torch_compile:
        model.decoder = torch.compile(  # type: ignore[assignment]
            model.decoder, dynamic=True, options={"shape_padding": True}
        )

    # TODO(balioglu): investigate!
    # The memory efficient SDPA implementation in PyTorch is not stable when
    # used with padded inputs.
    enable_memory_efficient_torch_sdpa(dp_model, False)

    log_model(dp_model, log)

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

    # Initialize the profiler.
    tb_dir = output_dir.joinpath("tb")

    profiler = Profiler(
        skip_first=15, active=3, log_dir=tb_dir, gang=root_gang, enabled=config.profile
    )

    # Set the seed for training.
    rng_bag.manual_seed(config.seed + dp_gang.rank)

    # Set up the finetuner.
    trainer = StandardTrainer[SequenceBatch](
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
        checkpoint_manager=checkpoint_manager,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        tb_dir=tb_dir,
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profiler=profiler,
        anomaly_detection=config.anomaly_detection,
        rng_bag=rng_bag,
        wall_watch=wall_watch,
    )

    trainer.maybe_restore()

    return trainer


@final
class InstructionFinetuneCriterion(AbstractCriterion[SequenceBatch]):
    """Computes cross entropy loss of a Language Model."""

    _train_metric_bag: SequenceModelMetricBag
    _valid_metric_bag: MetricBag

    def __init__(self, model: Module, gang: Gang) -> None:
        super().__init__(model)

        self._train_metric_bag = SequenceModelMetricBag(gang)

        self.register_non_stateful("_valid_metric_bag", MetricBag(gang))

    @override
    def compute_loss(self, batch: SequenceBatch) -> Tuple[Tensor, int]:
        batch, target_batch = as_auto_regressive_input(batch)

        output = cast(SequenceModelOutput, self._model(batch))

        loss = output.compute_loss(
            target_batch.seqs, loss_mask=target_batch.target_mask
        )

        if self._model.training:
            self._train_metric_bag.update(target_batch, loss)

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

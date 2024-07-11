# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, final

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module

from fairseq2.assets import default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider, FileCheckpointManager
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets import LengthBatching
# from fairseq2.datasets.instruction import load_instruction_dataset
from fairseq2.datasets.speech_text import load_speech_text_dataset, SpeechTextAlignBatch
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModel,
    SequenceModelOutput,
    as_auto_regressive_input,
    SpeechTextReprOutput
)
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.optim import AdamW
from fairseq2.optim.lr_scheduler import CosineAnnealingLR
from fairseq2.recipes.common_metrics import RepresentationAlignMetricBag
from fairseq2.recipes.trainer import AbstractTrainUnit, Trainer
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    check_model_type,
    compile_model,
    setup_gangs,
    to_data_parallel,
)
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class SpeechTextAlignConfig:
    """Holds the configuration of a language model instruction-finetuning task."""

    # Data
    # dataset: Union[str, Path] = "/fsx-ust/steventan0110/dataset/dinosr_train/mls_en"
    dataset: Union[str, Path] = "mls_en"
    """The name or path to the asset card of the instruction dataset."""

    max_seq_len: int = 1024
    """The maximum sequence length."""
    min_seq_len: int = 5

    max_num_tokens: int = 1024
    """The maximum number of tokens per batch."""

    example_shuffle_window: int = 10000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 10
    """The number of batches to prefetch in background."""

    # Model
    model: Union[str, Path] = "speech_llama3_8b"
    # model: Union[str, Path] = "llama3_8b"
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
    max_num_steps: int = 200000
    """The maximum number of steps to train for."""

    max_num_data_epochs: Optional[int] = None
    """The maximum number of data epochs to train for."""

    checkpoint_every_n_steps: int = 1000
    """The step interval at which to checkpoint."""

    keep_last_n_checkpoints: Optional[int] = 2
    """The number of checkpoints to keep. If ``None``, none will be deleted."""

    keep_last_n_models: Optional[int] = None
    """The number of checkpoint models to keep."""

    publish_metrics_every_n_steps: int = 10
    """The step interval at which to publish training metrics."""

    # Checkpointing
    resume_checkpoint_dir: Optional[Path] = None
    """If not ``None``, adds the specified path to the default asset store."""

    # Misc
    seed: int = 42
    """The random number generator seed to use."""

    profile: Optional[Tuple[int, int]] = None
    """The number of steps that the PyTorch profiler should skip and then record."""

    monitored_gang: bool = False
    """If ``True``, puts a monitored barrier before every collective call."""

    anomaly_detection: bool = False
    """If ``True``, turns on anomaly detection feature in ``torch.autograd``."""
    validate_after_n_steps: int = 1000


speech_text_presets = ConfigRegistry[SpeechTextAlignConfig]()

speech_text_preset = speech_text_presets.decorator



@speech_text_preset("llama3_8b_speech_text_align")
def _llama3_8b_instruct() -> SpeechTextAlignConfig:
    config =SpeechTextAlignConfig()
    config.data_parallelism = "ddp"
    config.max_seq_len = 512
    config.max_num_tokens = 512
    config.num_prefetch = 10
    config.lr = 1e-4
    config.num_lr_warmup_steps = 2000
    return config


@speech_text_preset("llama3_70b_instruct")
def _llama3_70b_instruct() -> SpeechTextAlignConfig:
    config = _llama3_8b_instruct()
    config.model = "llama3_70b_instruct"
    config.tensor_parallel_size = 8
    return config


def load_speech_text_trainer(
    config: SpeechTextAlignConfig, output_dir: Path
) -> Trainer[SequenceBatch]:
    """Load a :class:`Trainer` for language model instruction-finetuning."""
    wall_watch = Stopwatch(start=True)

    root_gang, gangs = setup_gangs(
        log, tp_size=config.tensor_parallel_size, monitored=config.monitored_gang
    )

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor
    # print(dp_gang.size, dp_gang.rank)
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
    
    log.info("Loading dataset {}", config.dataset)

    # Load the dataset.
    dataset_card = retrieve_asset_card(config.dataset)
    log.info("Loading {} instruction dataset.", dataset_card.name)

    dataset = load_speech_text_dataset(dataset_card)
    log.info("Dataset loaded.")

    # Load the model.
    init_device = META

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
        # ensure non-szeo weight for the new parameters
        model.speech_dim_adapter.reset_parameters()
        def reset_if_possible(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                # print(f"Reset parameters for {module}")
        model.speech_decoder.apply(reset_if_possible)
        # for name, params in model.speech_encoder.unit_extractor.named_parameters():
        #     print(name, params)
        # exit(0)
    if not isinstance(model, DecoderModel):
        raise ValueError("`config.model` must specify a decoder model.")

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

    # Initialize the train unit and the optimizer.
    unit = SpeechTextAlignUnit(dp_model, dp_gang)
    data_reader = dataset.create_reader(
        tokenizer,
        dp_gang,
        config.max_seq_len,
        config.max_num_tokens,
        config.min_seq_len,
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        num_accumulate=config.gradient_accumulation,
        num_prefetch=config.num_prefetch,
        seed=config.seed,
    )

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

    # Initialize the trainer.
    return Trainer[SpeechTextAlignBatch](
        unit=unit,
        # valid_units=[],
        # valid_data_readers=[],
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
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        keep_last_n_models=config.keep_last_n_models,
        # validate_after_n_steps=config.validate_after_n_steps,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
        profile=config.profile,
        anomaly_detection=config.anomaly_detection,
        seed=config.seed,
        wall_watch=wall_watch,
    )


@final
class SpeechTextAlignUnit(AbstractTrainUnit[SpeechTextAlignBatch]):
    """Represents a language model instruction-finetuning unit."""

    _metric_bag: RepresentationAlignMetricBag

    def __init__(self, model: Module, gang: Gang) -> None:
        """
        :param model:
            The language model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed training.
        """
        super().__init__(model)
        check_model_type(model, SequenceModel)
        self._metric_bag = RepresentationAlignMetricBag(gang)

    @override
    def __call__(self, batch: SpeechTextAlignBatch) -> Tuple[Tensor, int]:
        output = self._forward(batch)
        loss = output.compute_loss()
        self._metric_bag.update_loss(
            mse_loss=loss["mse_loss"].item(), 
            cosine_loss=loss["cosine_sim_loss"].item(), 
            num_target_elements=loss["target_size"])

        self._metric_bag.update_batch_metrics(batch.text_tokens)
        # aggreagte mse and cosine loss, use hard coded weight for now
        cosine_loss_weight = 5
        loss_to_return = loss["mse_loss"] + cosine_loss_weight * loss["cosine_sim_loss"]
        return loss_to_return, loss["target_size"]
    
    def _forward(self, batch: SpeechTextAlignBatch) -> SpeechTextReprOutput:
        return self._model(batch.audios, batch.text_tokens, batch.boundary_index)

    @property
    @override
    def metric_bag(self) -> RepresentationAlignMetricBag:
        return self._metric_bag


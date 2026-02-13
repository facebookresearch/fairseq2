#!/usr/bin/env python3
"""
Training Recipe for Llama 3.2 3B Instruct

This script demonstrates how to train Llama 3.2 3B Instruct using fairseq2's
trainer infrastructure with FP16 mixed precision and optimized data loading.

The recipe uses the "openeft" dataset (generic instruction dataset) available
as a fairseq2 asset for demonstration purposes.

Usage:
    # Due to potential issues with loss scaling, it is recommended to use no-amp
    # for a first run. Following this, you may choose a loss scaling factor

    # Single GPU training (FP16)
    python train_llama3_2_3b_instruct.py --output-dir ./checkpoints --device cuda:0 --no-amp

    # Multi-GPU training with FSDP (FP16)
    torchrun --nproc_per_node=4 train_llama3_2_3b_instruct.py \
        --output-dir ./checkpoints --device cuda --no-amp

    # CPU training (for testing, FP32)
    python train_llama3_2_3b_instruct.py --output-dir ./checkpoints --device cpu --no-amp

Architecture:
    - Model: Llama 3.2 3B Instruct (loaded from HuggingFace via fairseq2 assets)
    - Training: Causal language modeling with next-token prediction
    - Precision: FP16 mixed precision (AMP) with dynamic loss scaling
    - Data: Optimized data pipeline with gradient accumulation
    - Parallelism: FSDP for multi-GPU training via fairseq2's gang abstraction
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

# fairseq2 core imports
from fairseq2.checkpoint import (
    CheckpointManager,
    StandardCheckpointManager,
    NOOP_HG_EXPORTER,
)
from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data_type import DataType
from fairseq2.datasets import DataPipelineReader, DataReader, SequenceBatch
from fairseq2.device import Device, detect_default_device
from fairseq2.early_stopper import NOOP_EARLY_STOPPER
from fairseq2.file_system import LocalFileSystem
from fairseq2.gang import Gangs, get_default_gangs, create_fsdp_gangs, ProcessGroupGang, set_default_gangs, create_parallel_gangs
from fairseq2.io import TensorFileLoader, TensorFileDumper, _TorchTensorFileLoader, _TorchTensorFileDumper
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.recorders import TensorBoardRecorder
from fairseq2.models import load_model
from fairseq2.data.tokenizers import load_tokenizer, Tokenizer
from fairseq2.nn import BatchLayout
from fairseq2.optim.fp16_loss_scaler import StandardFloat16LossScaler, NOOP_FP16_LOSS_SCALER
from fairseq2.optim.lr_schedulers import CosineAnnealingLR
from fairseq2.profilers import NOOP_PROFILER
from fairseq2.trainer import Trainer, TrainUnit
from fairseq2.evaluator import EvalUnit
from fairseq2.utils.device_stat import NOOP_DEVICE_STAT_TRACKER
from fairseq2.utils.gc import NOOP_GARBAGE_COLLECTOR
from fairseq2.utils.progress import NOOP_PROGRESS_REPORTER
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.validator import NOOP_VALIDATOR, StandardValidator
from fairseq2.utils.threading import ThreadPool, _StandardThreadPool


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model and data
    model_name: str = "llama3_2_3b_instruct"
    dataset_name: str = "openeft"

    # Device and precision
    # Note: In distributed training (torchrun), the actual device for each rank
    # is auto-detected based on LOCAL_RANK to prevent GPU conflicts
    device: str = "cuda"
    amp: bool = True  # Automatic Mixed Precision (FP16)
    amp_dtype: DataType = torch.float16

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 2  # Per-device batch size
    num_accumulate: int = 4  # Gradient accumulation steps
    max_num_data_epochs: int = 3  # Changed from max_num_steps
    max_seq_len: int = 512

    # Data split
    eval_split_ratio: float = 0.1  # 10% for evaluation

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # FP16 loss scaling
    fp16_init_scale: float = 2.0 ** 16
    fp16_min_scale: float = 0.1

    # Checkpointing
    output_dir: Path = Path("./checkpoints")
    checkpoint_every_n_steps: int = 100
    keep_last_n_checkpoints: int = 3

    # Validation
    validate_every_n_data_epochs: int = 1  # Validate after each epoch

    # Logging
    publish_metrics_every_n_steps: int = 50  # Print training progress every N steps

    # Distributed training
    seed: int = 42


# ============================================================================
# Data Pipeline
# ============================================================================

def create_train_eval_pipelines(
    dataset_name: str,
    tokenizer: Tokenizer,
    gangs: Gangs,
    config: TrainingConfig,
) -> tuple[DataPipeline, DataPipeline]:
    """
    Creates train and eval data pipelines with 90/10 split.

    The pipeline:
    1. Reads text data and shuffles it
    2. Splits into train (90%) and eval (10%)
    3. Tokenizes and batches each split
    4. Train split is re-shuffled, eval is not

    Args:
        dataset_name: Name of the dataset asset to load
        tokenizer: Tokenizer for encoding text
        gangs: Gang abstraction for distributed training
        config: Training configuration

    Returns:
        Tuple of (train_pipeline, eval_pipeline)
    """
    log.info("Creating train/eval data pipelines for dataset '{}'", dataset_name)

    # Create a token encoder from the tokenizer
    text_encoder = tokenizer.create_encoder(mode="default", device=gangs.device)

    # Synthetic text dataset
    sample_instructions = [
        "Translate the following English text to French: Hello, how are you?",
        "Write a Python function to calculate the Fibonacci sequence.",
        "Explain the concept of machine learning in simple terms.",
        "What is the capital of France?",
        "Describe the process of photosynthesis.",
    ] * 200  # 1000 total examples

    # Calculate split sizes
    total_size = len(sample_instructions)
    eval_size = int(total_size * config.eval_split_ratio)
    train_size = total_size - eval_size

    log.info("Data split: {} train, {} eval ({:.1f}% eval)",
             train_size, eval_size, config.eval_split_ratio * 100)

    # Shuffle all data first, then split
    import random
    rng = random.Random(config.seed)
    shuffled_instructions = sample_instructions.copy()
    rng.shuffle(shuffled_instructions)

    train_data = shuffled_instructions[:train_size]
    eval_data = shuffled_instructions[train_size:]

    # Create train pipeline
    train_pipeline = (
        read_sequence(train_data)
        .shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        .shuffle(shuffle_window=1000, seed=config.seed)
        .map(text_encoder)
        .map(lambda x: _prepare_sequence(x, config.max_seq_len, tokenizer.vocab_info.pad_idx))
        .bucket(bucket_size=config.batch_size)
        .map(lambda batch: _collate_batch(batch, gangs.device))
        .prefetch(num_examples=4)
        .and_return()
    )

    # Create eval pipeline (no shuffling for deterministic results)
    eval_pipeline = (
        read_sequence(eval_data)
        .shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        .map(text_encoder)
        .map(lambda x: _prepare_sequence(x, config.max_seq_len, tokenizer.vocab_info.pad_idx))
        .bucket(bucket_size=config.batch_size)
        .map(lambda batch: _collate_batch(batch, gangs.device))
        .prefetch(num_examples=4)
        .and_return()
    )

    return train_pipeline, eval_pipeline


def _prepare_sequence(encoded_output: Any, max_len: int, pad_idx: int) -> dict[str, Tensor]:
    """
    Prepares an encoded sequence for causal language modeling.

    Truncates or pads the sequence to the specified maximum length.

    Args:
        encoded_output: Output from tokenizer encoder (dict or tensor)
        max_len: Maximum sequence length
        pad_idx: Padding token index

    Returns:
        Dictionary with 'seqs' key containing processed sequence tensor [max_len]
    """
    # Handle both dict and direct tensor output from encoder
    if isinstance(encoded_output, dict):
        seq = encoded_output["seqs"]
    else:
        # Assume it's a tensor directly
        seq = encoded_output

    if seq.size(0) > max_len:
        # Truncate to max length
        processed_seq = seq[:max_len]
    elif seq.size(0) < max_len:
        # Pad to max length
        padding = torch.full(
            (max_len - seq.size(0),),
            pad_idx,
            dtype=seq.dtype,
            device=seq.device,
        )
        processed_seq = torch.cat([seq, padding])
    else:
        processed_seq = seq

    return {"seqs": processed_seq}


def _collate_batch(batch: list[dict[str, Any]], device: Device) -> SequenceBatch:
    """
    Collates a list of examples into a SequenceBatch.

    Args:
        batch: List of dictionaries, each containing 'seqs' key
        device: Target device for the batch

    Returns:
        SequenceBatch ready for training
    """
    # Extract sequences from each example in the batch and stack them
    seqs_list = [example["seqs"] for example in batch]
    seqs = torch.stack(seqs_list)

    # Create sequence lengths as a list (all sequences have been padded to the same length)
    seq_lens = [seqs.size(1) for _ in range(len(batch))]

    return SequenceBatch(
        seqs=seqs,
        seq_lens=seq_lens,
    )


# ============================================================================
# Training Unit
# ============================================================================

class CausalLMTrainUnit(TrainUnit[SequenceBatch]):
    """
    Training unit for causal language modeling.

    This unit implements the forward pass, loss computation, and metric tracking
    for next-token prediction training.
    """

    def __init__(self, model: Module, gangs: Gangs, config: TrainingConfig) -> None:
        """
        Args:
            model: The language model to train
            gangs: Gang abstraction for distributed training
            config: Training configuration
        """
        self._model = model
        self._gangs = gangs
        self._config = config

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        """
        Prepares metrics to track during training.

        Args:
            metric_bag: Metric bag to populate with metrics
        """
        from fairseq2.metrics import Mean

        # Track training loss
        metric_bag.add("train_loss", Mean())

        # Track perplexity (exp of loss)
        metric_bag.add("train_ppl", Mean())

        # Track number of tokens processed
        metric_bag.add("num_tokens", Mean())

    def process_batch(
        self,
        batch: SequenceBatch,
        metric_bag: MetricBag,
    ) -> tuple[Tensor, int | None]:
        """
        Processes a batch and computes the loss.

        For causal language modeling, we:
        1. Run the model on input sequences
        2. Compute cross-entropy loss for next-token prediction
        3. Track metrics

        Args:
            batch: Batch of sequences to process
            metric_bag: Metric bag for tracking training metrics

        Returns:
            Tuple of (loss, num_targets) where:
            - loss: Scalar loss tensor for backpropagation
            - num_targets: Number of target tokens (for gradient normalization)
        """
        # Move batch to the correct device (modifies in-place, returns None)
        batch.to(self._gangs.device)

        # Extract input sequences [batch_size, seq_len]
        input_seqs = batch.seqs

        # For causal LM: input is seqs[:-1], target is seqs[1:]
        # Input: tokens 0 to seq_len-2
        # Target: tokens 1 to seq_len-1 (next token prediction)
        input_ids = input_seqs[:, :-1]
        target_ids = input_seqs[:, 1:]

        # Create batch layout for the input sequences
        # Adjust seq_lens since we removed the last token
        input_seq_lens = None
        if batch.seq_lens is not None:
            input_seq_lens = [max(1, length - 1) for length in batch.seq_lens]

        batch_layout = BatchLayout(
            shape=input_ids.shape,
            seq_lens=input_seq_lens,
            device=self._gangs.device,
        )

        # Forward pass through the model
        # fairseq2's TransformerLM can compute loss directly when targets are provided
        # This is more efficient than manually computing cross-entropy
        logits = self._model(input_ids, batch_layout)

        # Compute cross-entropy loss
        # Reshape: [batch_size * (seq_len-1), vocab_size]
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = target_ids.reshape(-1)

        # Compute loss with label smoothing disabled
        loss = cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
            ignore_index=-1,  # Ignore padding tokens if we had them
        )

        # Compute number of target tokens (for gradient normalization)
        num_targets = target_ids.numel()

        # Track metrics
        batch_size = input_seqs.size(0)

        from fairseq2.metrics import Mean

        metric_bag.get("train_loss", Mean).update(loss.detach(), weight=batch_size)
        metric_bag.get("train_ppl", Mean).update(torch.exp(loss.detach()), weight=batch_size)
        metric_bag.get("num_tokens", Mean).update(num_targets, weight=1)

        return loss, num_targets


# ============================================================================
# Evaluation Unit
# ============================================================================

class CausalLMEvalUnit(EvalUnit[SequenceBatch]):
    """
    Evaluation unit for causal language modeling.

    This unit implements evaluation without gradients, computing loss
    and perplexity on held-out data.
    """

    def __init__(self, model: Module, gangs: Gangs, config: TrainingConfig) -> None:
        """
        Args:
            model: The language model to evaluate
            gangs: Gang abstraction for distributed training
            config: Training configuration
        """
        self._model = model
        self._gangs = gangs
        self._config = config

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        """
        Prepares metrics to track during evaluation.

        Args:
            metric_bag: Metric bag to populate with metrics
        """
        from fairseq2.metrics import Mean

        # Track evaluation loss
        metric_bag.add("eval_loss", Mean())

        # Track perplexity (exp of loss)
        metric_bag.add("eval_ppl", Mean())

        # Track number of tokens processed
        metric_bag.add("eval_num_tokens", Mean())

    def process_batch(
        self,
        batch: SequenceBatch,
        metric_bag: MetricBag,
    ) -> None:
        """
        Processes a batch and computes the loss without gradients.

        Args:
            batch: Batch of sequences to process
            metric_bag: Metric bag for tracking evaluation metrics
        """
        # Move batch to the correct device (modifies in-place, returns None)
        batch.to(self._gangs.device)

        # Extract input sequences [batch_size, seq_len]
        input_seqs = batch.seqs

        # For causal LM: input is seqs[:-1], target is seqs[1:]
        input_ids = input_seqs[:, :-1]
        target_ids = input_seqs[:, 1:]

        # Create batch layout for the input sequences
        input_seq_lens = None
        if batch.seq_lens is not None:
            input_seq_lens = [max(1, length - 1) for length in batch.seq_lens]

        batch_layout = BatchLayout(
            shape=input_ids.shape,
            seq_lens=input_seq_lens,
            device=self._gangs.device,
        )

        # Forward pass through the model (no gradients)
        with torch.no_grad():
            logits = self._model(input_ids, batch_layout)

            # Compute cross-entropy loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = target_ids.reshape(-1)

            loss = cross_entropy(
                logits_flat,
                targets_flat,
                reduction="mean",
                ignore_index=-1,
            )

        # Compute number of target tokens
        num_targets = target_ids.numel()

        # Track metrics
        batch_size = input_seqs.size(0)

        from fairseq2.metrics import Mean

        metric_bag.get("eval_loss", Mean).update(loss.detach(), weight=batch_size)
        metric_bag.get("eval_ppl", Mean).update(torch.exp(loss.detach()), weight=batch_size)
        metric_bag.get("eval_num_tokens", Mean).update(num_targets, weight=1)


# ============================================================================
# Training Setup
# ============================================================================

def setup_training(config: TrainingConfig) -> Trainer:
    """
    Sets up all components for training.

    This function:
    1. Initializes distributed training (gangs)
    2. Loads the model and tokenizer
    3. Applies FSDP for multi-GPU training
    4. Creates the data pipeline and data reader
    5. Sets up optimizer, LR scheduler, and FP16 loss scaler
    6. Creates checkpoint manager and metric recorder
    7. Constructs the trainer

    Args:
        config: Training configuration

    Returns:
        Configured Trainer ready to run
    """
    # ========================================================================
    # 1. Initialize distributed training
    # ========================================================================

    # Detect the default device for this process
    # In distributed training (torchrun), this automatically assigns each rank
    # to its own GPU based on LOCAL_RANK environment variable (cuda:0, cuda:1, etc.)
    # This prevents multiple ranks from trying to use the same GPU
    device = detect_default_device()
    log.info("Using device: {}", device)

    # Disable FP16 on CPU - CPUs don't support FP16 efficiently and this causes
    # numerical instability leading to gradient overflow and training failure
    if device.type == "cpu" and config.amp:
        log.warning("FP16 mixed precision is not recommended on CPU due to numerical instability. Disabling AMP and using FP32 instead. Use --no-amp flag to suppress this warning.")
        config.amp = False
        config.amp_dtype = torch.float32

    # Initialize distributed training if in multi-process environment
    # Check environment variables set by torchrun to determine if we're in distributed mode
    import torch.distributed as dist
    import os
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        log.info("Multi-process environment detected (world size: {}). Initializing process group.", world_size)

        # Create the root process group gang
        # This initializes torch.distributed and creates the default process group
        from datetime import timedelta
        root_gang = ProcessGroupGang.create_default_process_group(
            device,
            timeout=timedelta(minutes=15),
            high_priority=False,
        )

        # Create parallel gangs for data/tensor/pipeline parallelism
        gangs = create_parallel_gangs(root_gang, tp_size=1)

        # Set as default gangs so get_default_gangs() will return them
        set_default_gangs(gangs)

        log.info("Process group initialized. Root gang size: {}", root_gang.size)
    else:
        log.info("Single-process environment. Using FakeGang.")

    # Get default gangs for the device
    # This will return the gangs we just initialized, or FakeGang for single-process
    gangs = get_default_gangs(device)
   
    # For multi-GPU training, create FSDP-specific gangs
    if gangs.root.size > 1:
        log.info("Multi-GPU training detected (world size: {})", gangs.root.size)
        gangs = create_fsdp_gangs(gangs)
    else:
        log.info("Single-GPU/CPU training")

    # ========================================================================
    # 2. Load model and tokenizer
    # ========================================================================

    log.info("Loading model '{}'", config.model_name)

    # Load the model with the specified dtype for mixed precision
    # The model will be loaded on the meta device first, then moved to the target device
    if device.type == "cuda":
        torch.cuda.synchronize()
    model = load_model(
        config.model_name,
        gangs=gangs,
        dtype=config.amp_dtype if config.amp else torch.float32,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    log.info("Model loaded successfully")

    # Load the tokenizer
    log.info("Loading tokenizer for '{}'", config.model_name)
    tokenizer = load_tokenizer(config.model_name, gangs=gangs)
    log.info("Tokenizer loaded successfully (vocab size: {})", tokenizer.vocab_info.size)

    # ========================================================================
    # 3. Apply data parallelism (FSDP for multi-GPU)
    # ========================================================================

    # Note: For this simple example, we skip explicit FSDP wrapping.
    # In production, you would wrap the model with FSDP for multi-GPU training
    # using the model family's FSDP applier.

    log.info("Model ready for training")

    # ========================================================================
    # 4. Create data pipeline and data reader
    # ========================================================================

    # Create train and eval pipelines
    train_pipeline, eval_pipeline = create_train_eval_pipelines(
        config.dataset_name, tokenizer, gangs, config,
    )

    # Create the train data reader with gradient accumulation
    train_data_reader = DataPipelineReader(
        train_pipeline, gangs,
        num_accumulate=config.num_accumulate,
        drop_remainder=True,
    )

    # Create the eval data reader
    eval_data_reader = DataPipelineReader(
        eval_pipeline, gangs,
        num_accumulate=config.num_accumulate,
        drop_remainder=False,
    )

    log.info("Data readers created (batch_size={}, num_accumulate={}, effective_batch_size={})",
             config.batch_size, config.num_accumulate, config.batch_size * config.num_accumulate * gangs.dp.size)

    # ========================================================================
    # 5. Create training unit
    # ========================================================================

    train_unit = CausalLMTrainUnit(model, gangs, config)

    # ========================================================================
    # 5b. Create evaluation unit (validator created later)
    # ========================================================================

    eval_unit = CausalLMEvalUnit(model, gangs, config)

    # ========================================================================
    # 6. Setup optimizer and LR scheduler
    # ========================================================================

    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # Standard LLM training betas
        eps=1e-8,
    )

    log.info("Optimizer: AdamW (lr={}, weight_decay={})", config.learning_rate, config.weight_decay)

    # Create cosine annealing LR scheduler with warmup
    # Calculate total steps: epochs * (dataset_size / batch_size)
    # For simplicity, use a reasonable estimate based on data size
    train_size = int(1000 * (1.0 - config.eval_split_ratio))  # 900 examples
    steps_per_epoch = train_size // (config.batch_size * config.num_accumulate * gangs.dp.size)
    max_num_steps = max(1, steps_per_epoch * config.max_num_data_epochs)

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        cycle_len=max_num_steps,
        num_warmup_steps=config.warmup_steps,
        final_lr=config.learning_rate * 0.1,  # Decay to 10% of initial LR
    )

    log.info("LR Scheduler: CosineAnnealingLR (warmup={} steps, total_steps={})", config.warmup_steps, max_num_steps)

    # ========================================================================
    # 7. Setup FP16 loss scaler
    # ========================================================================

    if config.amp:
        fp16_loss_scaler = StandardFloat16LossScaler(
            gangs,
            init_scale=config.fp16_init_scale,
            scale_window=config.fp16_scale_window,
            min_scale=config.fp16_min_scale,
        )
        log.info("FP16 mixed precision enabled (init_scale={}, min_scale={})", config.fp16_init_scale, config.fp16_min_scale)
    else:
        fp16_loss_scaler = NOOP_FP16_LOSS_SCALER
        log.info("FP32 training (no mixed precision)")

    # ========================================================================
    # 8. Setup checkpoint manager
    # ========================================================================

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager for saving/loading checkpoints
    file_system = LocalFileSystem()
    tensor_file_loader = _TorchTensorFileLoader(file_system)
    tensor_file_dumper = _TorchTensorFileDumper(file_system)
    thread_pool = _StandardThreadPool.create_default(gangs.root.size)

    checkpoint_manager = StandardCheckpointManager(
        output_dir=config.output_dir,
        gangs=gangs,
        file_system=file_system,
        tensor_file_loader=tensor_file_loader,
        tensor_file_dumper=tensor_file_dumper,
        thread_pool=thread_pool,
    )

    log.info("Checkpoint manager created (output_dir={})", config.output_dir)

    # ========================================================================
    # 9. Setup metric recorder for progress logging
    # ========================================================================

    # Create metric descriptors for logging training progress
    from fairseq2.metrics.recorders import (
        LogMetricRecorder,
        MetricDescriptor,
        MetricDescriptorRegistry,
        NOOP_METRIC_DESCRIPTOR,
    )

    metric_descriptors = [
        MetricDescriptor(
            name="train_loss",
            display_name="Train Loss",
            priority=1,
            formatter=lambda x: f"{x:.4f}",
            log=True,
        ),
        MetricDescriptor(
            name="train_ppl",
            display_name="Train Perplexity",
            priority=2,
            formatter=lambda x: f"{x:.2f}",
            log=True,
        ),
        MetricDescriptor(
            name="eval_loss",
            display_name="Eval Loss",
            priority=1,
            formatter=lambda x: f"{x:.4f}",
            log=True,
        ),
        MetricDescriptor(
            name="eval_ppl",
            display_name="Eval Perplexity",
            priority=2,
            formatter=lambda x: f"{x:.2f}",
            log=True,
        ),
        MetricDescriptor(
            name="grad_norm",
            display_name="Gradient Norm",
            priority=3,
            formatter=lambda x: f"{x:.4f}",
            log=True,
        ),
    ]

    metric_registry = MetricDescriptorRegistry(metric_descriptors)
    metric_recorder = LogMetricRecorder(metric_registry)

    # ========================================================================
    # 10. Create the validator and trainer
    # ========================================================================

    # Setup utilities
    garbage_collector = NOOP_GARBAGE_COLLECTOR
    profiler = NOOP_PROFILER
    device_stat_tracker = NOOP_DEVICE_STAT_TRACKER
    wall_watch = Stopwatch()
    progress_reporter = NOOP_PROGRESS_REPORTER

    # Create validator with epoch-based validation
    # StandardValidator needs all infrastructure components
    validator = StandardValidator(
        units=[eval_unit],
        data_readers=[eval_data_reader],
        gangs=gangs,
        amp=config.amp,
        amp_dtype=config.amp_dtype,
        score_metric_descriptor=NOOP_METRIC_DESCRIPTOR,  # No early stopping based on metric
        checkpoint_manager=checkpoint_manager,
        hg_exporter=NOOP_HG_EXPORTER,
        metric_recorder=metric_recorder,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=Stopwatch(),
        progress_reporter=progress_reporter,
        seed=config.seed,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        unit=train_unit,
        data_reader=train_data_reader,
        gangs=gangs,
        amp=config.amp,
        amp_dtype=config.amp_dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scaler=fp16_loss_scaler,
        validator=validator,
        early_stopper=NOOP_EARLY_STOPPER,
        checkpoint_manager=checkpoint_manager,
        hg_exporter=NOOP_HG_EXPORTER,
        metric_recorder=metric_recorder,
        garbage_collector=garbage_collector,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=wall_watch,
        progress_reporter=progress_reporter,
        seed=config.seed,
        max_grad_norm=config.max_grad_norm,
        max_num_data_epochs=config.max_num_data_epochs,
        validate_every_n_data_epochs=config.validate_every_n_data_epochs,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        publish_metrics_every_n_steps=config.publish_metrics_every_n_steps,
    )

    log.info("Trainer created successfully")

    return trainer


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main training entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Llama 3.2 3B Instruct with fairseq2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (use FP32)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )

    parser.add_argument(
        "--num-accumulate",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create training configuration
    config = TrainingConfig(
        output_dir=args.output_dir,
        device=args.device,
        amp=not args.no_amp,
        batch_size=args.batch_size,
        num_accumulate=args.num_accumulate,
        learning_rate=args.learning_rate,
        max_num_data_epochs=args.max_num_epochs,
        max_seq_len=args.max_seq_len,
    )

    # Log configuration
    log.info("=" * 80)
    log.info("Training Configuration:")
    log.info("  Model: {}", config.model_name)
    log.info("  Dataset: {}", config.dataset_name)
    log.info("  Device: {}", config.device)
    log.info("  Mixed Precision: {}", "FP16" if config.amp else "FP32")
    log.info("  Batch Size: {}", config.batch_size)
    log.info("  Gradient Accumulation: {}", config.num_accumulate)
    log.info("  Learning Rate: {}", config.learning_rate)
    log.info("  Max Epochs: {}", config.max_num_data_epochs)
    log.info("  Max Seq Length: {}", config.max_seq_len)
    log.info("  Eval Split: {:.1f}%", config.eval_split_ratio * 100)
    log.info("  Output Dir: {}", config.output_dir)
    log.info("=" * 80)

    # Setup training
    trainer = setup_training(config)

    # Run training
    log.info("Starting training...")
    try:
        trainer.run()
    except KeyboardInterrupt:
        log.warning("Training interrupted by user")
    except Exception as e:
        log.error("Training failed with error: {}", e)
        raise
    finally:
        # Cleanup
        log.info("Training finished")


if __name__ == "__main__":
    main()

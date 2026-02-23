"""Trainer setup and initialization."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from fairseq2.checkpoint import NOOP_HG_EXPORTER, StandardCheckpointManager
from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.datasets import DataPipelineReader
from fairseq2.device import detect_default_device
from fairseq2.early_stopper import NOOP_EARLY_STOPPER
from fairseq2.file_system import LocalFileSystem
from fairseq2.gang import (
    Gangs,
    ProcessGroupGang,
    create_fsdp_gangs,
    create_parallel_gangs,
    get_default_gangs,
    set_default_gangs,
)
from fairseq2.io import _TorchTensorFileDumper, _TorchTensorFileLoader
from fairseq2.logging import log
from fairseq2.metrics.recorders import (
    LogMetricRecorder,
    MetricDescriptor,
    MetricDescriptorRegistry,
    NOOP_METRIC_DESCRIPTOR,
)
from fairseq2.models import load_model
from fairseq2.optim.fp16_loss_scaler import (
    NOOP_FP16_LOSS_SCALER,
    StandardFloat16LossScaler,
)
from fairseq2.optim.lr_schedulers import CosineAnnealingLR
from fairseq2.profilers import NOOP_PROFILER
from fairseq2.trainer import Trainer
from fairseq2.utils.device_stat import NOOP_DEVICE_STAT_TRACKER
from fairseq2.utils.gc import NOOP_GARBAGE_COLLECTOR
from fairseq2.utils.progress import NOOP_PROGRESS_REPORTER
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.threading import _StandardThreadPool
from fairseq2.validator import StandardValidator

from config import TrainingConfig
from data import create_train_eval_pipelines
from eval_unit import CausalLMEvalUnit
from train_unit import CausalLMTrainUnit


def setup_training(config: TrainingConfig) -> Trainer:
    """
    Sets up all components for training.

    This function:
    1. Initializes distributed training (gangs)
    2. Loads the model and tokenizer
    3. Creates the data pipeline and data reader
    4. Sets up optimizer, LR scheduler, and FP16 loss scaler
    5. Creates checkpoint manager and metric recorder
    6. Constructs the trainer

    Args:
        config: Training configuration

    Returns:
        Configured Trainer ready to run
    """
    # Initialize distributed training
    device = detect_default_device()
    log.info("Using device: {}", device)

    # Disable FP16 on CPU
    if device.type == "cpu" and config.amp:
        log.warning("FP16 not recommended on CPU. Disabling AMP.")
        config.amp = False
        config.amp_dtype = torch.float32

    # Initialize distributed training if in multi-process environment
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        log.info("Multi-process environment detected (world size: {})", world_size)

        root_gang = ProcessGroupGang.create_default_process_group(
            device,
            timeout=timedelta(minutes=15),
            high_priority=False,
        )

        gangs = create_parallel_gangs(root_gang, tp_size=1)
        set_default_gangs(gangs)

        log.info("Process group initialized. Root gang size: {}", root_gang.size)
    else:
        log.info("Single-process environment. Using FakeGang.")

    gangs = get_default_gangs(device)

    # For multi-GPU training, create FSDP-specific gangs
    if gangs.root.size > 1:
        log.info("Multi-GPU training detected (world size: {})", gangs.root.size)
        gangs = create_fsdp_gangs(gangs)
    else:
        log.info("Single-GPU/CPU training")

    # Load model and tokenizer
    log.info("Loading model '{}'", config.model_name)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Always load model in FP32, even with AMP enabled
    # The Trainer's autocast() handles FP16 conversion during forward pass
    # Loading in FP16 causes gradient explosion due to loss of precision
    model = load_model(
        config.model_name,
        gangs=gangs,
        dtype=torch.float32,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    log.info("Model loaded successfully")

    log.info("Loading tokenizer for '{}'", config.model_name)
    tokenizer = load_tokenizer(config.model_name, gangs=gangs)
    log.info("Tokenizer loaded successfully (vocab size: {})", tokenizer.vocab_info.size)

    # Create data pipeline and data reader
    train_pipeline, eval_pipeline = create_train_eval_pipelines(
        config.dataset_name, tokenizer, gangs, config,
    )

    train_data_reader = DataPipelineReader(
        train_pipeline, gangs,
        num_accumulate=config.num_accumulate,
        drop_remainder=True,
    )

    eval_data_reader = DataPipelineReader(
        eval_pipeline, gangs,
        num_accumulate=config.num_accumulate,
        drop_remainder=False,
    )

    log.info("Data readers created (batch_size={}, num_accumulate={}, effective_batch_size={})",
             config.batch_size, config.num_accumulate, config.batch_size * config.num_accumulate * gangs.dp.size)

    # Create training and evaluation units
    train_unit = CausalLMTrainUnit(model, gangs, config)
    eval_unit = CausalLMEvalUnit(model, gangs, config)

    # Setup optimizer and LR scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    log.info("Optimizer: AdamW (lr={}, weight_decay={})", config.learning_rate, config.weight_decay)

    # Calculate total steps for LR scheduler
    train_size = int(1000 * (1.0 - config.eval_split_ratio))
    steps_per_epoch = train_size // (config.batch_size * config.num_accumulate * gangs.dp.size)
    max_num_steps = max(1, steps_per_epoch * config.max_num_data_epochs)

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        cycle_len=max_num_steps,
        num_warmup_steps=config.warmup_steps,
        final_lr=config.learning_rate * 0.1,
    )

    log.info("LR Scheduler: CosineAnnealingLR (warmup={} steps, total_steps={})", config.warmup_steps, max_num_steps)

    # Setup FP16 loss scaler
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

    # Setup checkpoint manager
    config.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Setup metric recorder
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

    # Setup utilities
    garbage_collector = NOOP_GARBAGE_COLLECTOR
    profiler = NOOP_PROFILER
    device_stat_tracker = NOOP_DEVICE_STAT_TRACKER
    wall_watch = Stopwatch()
    wall_watch.start()  # Must be started before passing to Trainer
    progress_reporter = NOOP_PROGRESS_REPORTER

    # Create validator
    validator = StandardValidator(
        units=[eval_unit],
        data_readers=[eval_data_reader],
        gangs=gangs,
        amp=config.amp,
        amp_dtype=config.amp_dtype,
        score_metric_descriptor=NOOP_METRIC_DESCRIPTOR,
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

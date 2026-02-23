"""Training configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model and data
    model_name: str = "llama3_2_3b_instruct"
    dataset_name: str = "openeft"

    # Device and precision
    device: str = "cuda"
    amp: bool = True
    amp_dtype: torch.dtype = torch.float16

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_accumulate: int = 4
    max_num_data_epochs: int = 3
    max_seq_len: int = 512

    # Data split
    eval_split_ratio: float = 0.1
    seed: int = 42

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # FP16 loss scaling
    fp16_init_scale: float = 2.0 ** 16
    fp16_min_scale: float = 0.1
    fp16_scale_window: int = 1000

    # Checkpointing
    output_dir: Path = Path("./checkpoints")
    checkpoint_every_n_steps: int = 100
    keep_last_n_checkpoints: int = 3

    # Validation
    validate_every_n_data_epochs: int = 1

    # Logging
    publish_metrics_every_n_steps: int = 50

#!/usr/bin/env python3
"""
Training Recipe for Llama 3.2 3B Instruct

This script demonstrates how to train Llama 3.2 3B Instruct using fairseq2's
trainer infrastructure with FP16 mixed precision and optimized data loading.

Usage:
    # Single GPU training (FP16)
    python train.py --config config.yaml --device cuda:0

    # Multi-GPU training with FSDP (FP16)
    torchrun --nproc_per_node=4 train.py --config config.yaml --device cuda

    # CPU training (for testing, FP32)
    python train.py --config config.yaml --device cpu --no-amp

Architecture:
    - Model: Llama 3.2 3B Instruct (loaded from HuggingFace via fairseq2 assets)
    - Training: Causal language modeling with next-token prediction
    - Precision: FP16 mixed precision (AMP) with dynamic loss scaling
    - Data: Optimized data pipeline with gradient accumulation
    - Parallelism: FSDP for multi-GPU training via fairseq2's gang abstraction
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from fairseq2.logging import log

from config import TrainingConfig
from trainer import setup_training


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_config_from_yaml(config_dict: dict, args: argparse.Namespace) -> TrainingConfig:
    """Create TrainingConfig from YAML dictionary and command-line args."""

    # Parse dtype string to torch dtype
    dtype_str = config_dict["precision"]["amp_dtype"]
    amp_dtype = getattr(torch, dtype_str)

    return TrainingConfig(
        # Model and data
        model_name=config_dict["model"]["name"],
        dataset_name=config_dict["dataset"]["name"],

        # Device and precision (allow CLI override)
        device=args.device if args.device else config_dict["device"]["type"],
        amp=not args.no_amp if args.no_amp is not None else config_dict["precision"]["amp"],
        amp_dtype=amp_dtype,

        # Training hyperparameters (allow CLI overrides)
        learning_rate=args.learning_rate if args.learning_rate else config_dict["training"]["learning_rate"],
        batch_size=args.batch_size if args.batch_size else config_dict["training"]["batch_size"],
        num_accumulate=args.num_accumulate if args.num_accumulate else config_dict["training"]["num_accumulate"],
        max_num_data_epochs=args.max_num_epochs if args.max_num_epochs else config_dict["training"]["max_num_data_epochs"],
        max_seq_len=args.max_seq_len if args.max_seq_len else config_dict["training"]["max_seq_len"],
        weight_decay=config_dict["training"]["weight_decay"],
        max_grad_norm=config_dict["training"]["max_grad_norm"],
        warmup_steps=config_dict["training"]["warmup_steps"],

        # Data split
        eval_split_ratio=config_dict["data"]["eval_split_ratio"],
        seed=config_dict["data"]["seed"],

        # FP16 loss scaling
        fp16_init_scale=config_dict["fp16"]["init_scale"],
        fp16_min_scale=config_dict["fp16"]["min_scale"],
        fp16_scale_window=config_dict["fp16"]["scale_window"],

        # Checkpointing (allow CLI override)
        output_dir=Path(args.output_dir) if args.output_dir else Path(config_dict["checkpoint"]["output_dir"]),
        checkpoint_every_n_steps=config_dict["checkpoint"]["checkpoint_every_n_steps"],
        keep_last_n_checkpoints=config_dict["checkpoint"]["keep_last_n_checkpoints"],

        # Validation
        validate_every_n_data_epochs=config_dict["validation"]["validate_every_n_data_epochs"],

        # Logging
        publish_metrics_every_n_steps=config_dict["logging"]["publish_metrics_every_n_steps"],
    )


def main() -> None:
    """Main training entry point."""

    parser = argparse.ArgumentParser(
        description="Train Llama 3.2 3B Instruct with fairseq2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to configuration YAML file",
    )

    # Optional overrides for common parameters
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--no-amp", action="store_true", default=None, help="Disable automatic mixed precision")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--num-accumulate", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-num-epochs", type=int, default=None, help="Maximum number of training epochs")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration from YAML
    config_dict = load_config(args.config)

    # Create training configuration
    config = create_config_from_yaml(config_dict, args)

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
        log.info("Training finished")


if __name__ == "__main__":
    main()

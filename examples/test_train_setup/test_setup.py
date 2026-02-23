#!/usr/bin/env python3
"""
Quick test to verify training setup completes without errors.
This doesn't run actual training, just verifies all components are created correctly.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

# Add parent directory to path to import train_llama module
sys.path.insert(0, str(Path(__file__).parent.parent / "train_llama"))

from config import TrainingConfig
from trainer import setup_training
from fairseq2.logging import log


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Test training setup for fairseq2"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration
    config_dict = load_config(args.config)

    # Create TrainingConfig from YAML
    config = TrainingConfig(
        output_dir=Path(config_dict["training"]["output_dir"]),
        device=config_dict["training"]["device"],
        amp=config_dict["training"]["amp"],
        batch_size=config_dict["training"]["batch_size"],
        num_accumulate=config_dict["training"]["num_accumulate"],
        learning_rate=config_dict["training"]["learning_rate"],
        max_num_data_epochs=config_dict["training"]["max_num_data_epochs"],
        max_seq_len=config_dict["training"]["max_seq_len"],
    )

    log.info("=" * 80)
    log.info("Testing Training Setup")
    log.info("=" * 80)

    try:
        # Setup training components
        trainer = setup_training(config)

        log.info("=" * 80)
        log.info("SUCCESS: All components created successfully!")
        log.info("  - Model: loaded")
        log.info("  - Tokenizer: loaded")
        log.info("  - Train pipeline: created")
        log.info("  - Eval pipeline: created")
        log.info("  - Train unit: created")
        log.info("  - Eval unit: created")
        log.info("  - Validator: created")
        log.info("  - Trainer: created")
        log.info("=" * 80)

        return 0
    except Exception as e:
        log.error("FAILED: Setup failed with error: {}", e)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup test checkpoints
        if Path(config_dict["training"]["output_dir"]).exists():
            shutil.rmtree(config_dict["training"]["output_dir"])


if __name__ == "__main__":
    sys.exit(main())

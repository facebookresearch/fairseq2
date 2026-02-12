#!/usr/bin/env python3
"""
Quick test to verify training setup completes without errors.
This doesn't run actual training, just verifies all components are created correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from train_llama3_2_3b_instruct import TrainingConfig, setup_training
from fairseq2.logging import log
import logging

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create test config
    config = TrainingConfig(
        output_dir=Path("./test_checkpoints"),
        device="cpu",
        amp=False,  # Disable AMP for CPU
        batch_size=2,
        num_accumulate=4,
        learning_rate=1e-5,
        max_num_data_epochs=1,
        max_seq_len=512,
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
        import shutil
        if Path("./test_checkpoints").exists():
            shutil.rmtree("./test_checkpoints")

if __name__ == "__main__":
    sys.exit(main())

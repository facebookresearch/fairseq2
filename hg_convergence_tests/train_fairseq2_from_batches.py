#!/usr/bin/env python3
"""Train fairseq2 HG model using pre-extracted batches (for convergence testing).

This script loads the same pre-extracted batches that HuggingFace uses,
ensuring both frameworks train on identical data for fair comparison.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# Enable full determinism BEFORE any CUDA operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Disable Flash Attention for determinism
os.environ['DISABLE_FLASH_ATTN'] = '1'

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add fairseq2 to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairseq2.models.hg_qwen_omni.api import load_hg_model_simple
from fairseq2.models.hg_qwen_omni.tokenizer import load_hg_tokenizer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PreloadedDataset(Dataset):
    """Dataset that loads pre-extracted batches."""

    def __init__(self, data_dir: Path, num_batches: int = 100):
        self.data_dir = data_dir
        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch_path = self.data_dir / f"batch_{idx:04d}.pt"
        batch = torch.load(batch_path)
        # Squeeze to remove batch dim since DataLoader will add it back
        return {
            "input_ids": batch["input_ids"].squeeze(0),
            "attention_mask": batch["attention_mask"].squeeze(0),
            "labels": batch["labels"].squeeze(0),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fairseq2 HG model with pre-extracted batches"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with extracted batches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save checkpoint",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of training steps",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed (default: 2)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    print(f"Training fairseq2 HG model for {args.num_steps} steps")
    print(f"Using pre-extracted batches from: {args.data_dir}")
    print(f"Model: {args.model_name}")
    print(f"Random seed: {args.seed}")

    # Load model via fairseq2 HG adapter
    print(f"\nLoading model via fairseq2 HG adapter...")
    wrapped_model = load_hg_model_simple(
        args.model_name,
        model_type="causal_lm",
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",  # Disable Flash Attention for determinism
    )

    # Extract the underlying HuggingFace model from the adapter
    # The adapter provides a .hf_model property to access the wrapped model
    if hasattr(wrapped_model, "hf_model"):
        model = wrapped_model.hf_model
        print("Extracted underlying HuggingFace model from fairseq2 adapter")
    else:
        model = wrapped_model

    print("Model loaded successfully (Flash Attention disabled for determinism)")

    # Move to GPU
    device = torch.device("cuda:0")
    model = model.to(device)

    # Set to eval mode to disable dropout for deterministic training
    # This ensures exact reproducibility while still allowing gradient updates
    model.eval()
    print("Model set to eval() mode for deterministic training (dropout disabled)")

    # Optimizer (matching HuggingFace exactly)
    from torch.optim import AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    # LR Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
        eta_min=6e-5,
    )

    # Load dataset
    num_batches_per_epoch = 100
    dataset = PreloadedDataset(args.data_dir, num_batches=num_batches_per_epoch)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Calculate epochs
    num_epochs = (args.num_steps + num_batches_per_epoch - 1) // num_batches_per_epoch
    print(f"Dataset: {num_batches_per_epoch} batches")
    print(f"Training for {num_epochs} epochs to reach {args.num_steps} steps")

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.num_steps:
                break

            # Move batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {global_step}/{args.num_steps}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )

            global_step += 1

        if global_step >= args.num_steps:
            break

    # Save checkpoint
    print("Saving checkpoint...")
    checkpoint_path = args.output_dir / f"checkpoint_{args.num_steps}.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

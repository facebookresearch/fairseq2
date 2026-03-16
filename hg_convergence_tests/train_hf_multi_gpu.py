#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train with HuggingFace Transformers using extracted fairseq2 batches (multi-GPU with DDP)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable full determinism BEFORE any CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Disable Flash Attention for determinism
os.environ["DISABLE_FLASH_ATTN"] = "1"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Not using distributed training
        return 0, 1, 0

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


class PreloadedDataset(Dataset):
    """Dataset that loads pre-extracted batches from fairseq2."""

    def __init__(self, data_dir: Path, num_batches: int = 200):
        self.data_dir = data_dir
        self.num_batches = num_batches
        self.access_log = []  # Track which indices are accessed

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.access_log.append(idx)  # Track access
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
        description="Train with Transformers (multi-GPU with DDP)"
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
        default=10000,
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
        help="Random seed (default: 2, matching fairseq2)",
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(args.seed)

    if rank == 0:
        print(f"Training on {world_size} GPUs for {args.num_steps} steps")
        print("Using FULL FINE-TUNING (not LoRA)")
        print("Mixed precision: bfloat16 (matching fairseq2)")
        print(f"Random seed: {args.seed}")

    # Load model
    if rank == 0:
        print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,  # Match fairseq2's dtype
        local_files_only=True,  # Use cached files
        trust_remote_code=True,
        attn_implementation="eager",  # Disable Flash Attention for determinism
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        trust_remote_code=True,
    )
    if rank == 0:
        print("Model loaded successfully (Flash Attention disabled for determinism)")

    # Move to GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Wrap with DDP (use DDP instead of FSDP since model fits on single GPU;
    # FSDP breaks Gemma 3's weight tying between lm_head and embeddings)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"Model wrapped with DDP across {world_size} GPUs")

    # Set to eval mode to disable dropout for deterministic training
    model.eval()
    if rank == 0:
        print("Model set to eval() mode for deterministic training (dropout disabled)")

    # Get model parameters
    model_params = model.parameters()

    # Optimizer (matching fairseq2 exactly)
    optimizer = AdamW(
        model_params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,  # Match fairseq2's fused AdamW
    )

    # LR Scheduler (matching fairseq2's cosine annealing)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,  # cycle_len = num_steps
        eta_min=6e-5,  # final_lr = 3e-4 * 0.2
    )

    # Load dataset
    num_batches_per_epoch = 200
    dataset = PreloadedDataset(args.data_dir, num_batches=num_batches_per_epoch)

    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Keep original order for reproducibility
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    # Calculate epochs
    steps_per_epoch = num_batches_per_epoch // world_size
    num_epochs = (args.num_steps + steps_per_epoch - 1) // steps_per_epoch
    if rank == 0:
        print(f"Dataset: {num_batches_per_epoch} batches")
        print(f"Steps per epoch per GPU: {steps_per_epoch}")
        print(f"Training for {num_epochs} epochs to reach {args.num_steps} steps")

    # Training loop with multiple epochs
    global_step = 0

    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

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

            # Gradient clipping (matching fairseq2's max_grad_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging (only rank 0)
            if rank == 0 and global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {global_step}/{args.num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )

            global_step += 1

        # Log batch indices processed in first epoch for verification (rank 0 only)
        if epoch == 0 and rank == 0:
            print(
                f"[HF Multi-GPU] Epoch 0 batch indices on rank 0: {sorted(dataset.access_log)}"
            )
            dataset.access_log.clear()  # Clear for next epoch

        if global_step >= args.num_steps:
            break

    # Save checkpoint (only rank 0)
    if rank == 0:
        print("Saving checkpoint...")
        checkpoint_path = args.output_dir / f"checkpoint_{args.num_steps}.pt"

        # For DDP, unwrap with .module to get the original model state dict
        if world_size > 1:
            state_dict = model.module.state_dict()
            torch.save({"model": state_dict}, checkpoint_path)
        else:
            torch.save({"model": model.state_dict()}, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

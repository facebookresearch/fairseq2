#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train with Unsloth using extracted fairseq2 batches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from unsloth import FastLanguageModel


class PreloadedDataset(Dataset):
    """Dataset that loads pre-extracted batches from fairseq2."""

    def __init__(self, data_dir: Path, num_batches: int = 100):
        self.data_dir = data_dir
        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch_path = self.data_dir / f"batch_{idx:04d}.pt"
        batch = torch.load(batch_path)
        return {
            "input_ids": batch["input_ids"].squeeze(0),  # Remove batch dim
            "attention_mask": batch["attention_mask"].squeeze(0),
            "labels": batch["labels"].squeeze(0),
        }


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank, dist.get_rank(), dist.get_world_size()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with Unsloth")
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
        default=100,
        help="Number of training steps",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name",
    )

    args = parser.parse_args()

    # Setup distributed
    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training with {world_size} GPUs for {args.num_steps} steps")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # No quantization for fair comparison
    )

    # Move model to GPU and wrap with DDP
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer (matching fairseq2 exactly)
    optimizer = AdamW(
        model.parameters(),
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
        eta_min=6e-5,          # final_lr = 3e-4 * 0.2
    )

    # Load dataset
    dataset = PreloadedDataset(args.data_dir, num_batches=args.num_steps)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # Keep original order
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
    )

    # Training loop
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break

        # Move batch to GPU
        input_ids = batch["input_ids"].to(local_rank)
        attention_mask = batch["attention_mask"].to(local_rank)
        labels = batch["labels"].to(local_rank)

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

        # Logging
        if rank == 0 and step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{args.num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    # Save checkpoint (only rank 0)
    if rank == 0:
        checkpoint_path = args.output_dir / "checkpoint_100.pt"
        # Unwrap DDP and save state dict
        state_dict = model.module.state_dict()
        torch.save({"model": state_dict}, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

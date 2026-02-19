# fairseq2 vs Unsloth Convergence Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use 10x-engineer:executing-plans to implement this plan task-by-task.

**Goal:** Build a parallel training comparison system that validates fairseq2's HuggingFace adapter produces identical convergence to Unsloth.

**Architecture:** Extract batches from fairseq2's data pipeline to disk, run parallel training on 2x4 GPUs (fairseq2 and Unsloth), then compare checkpoints using np.allclose with bf16-appropriate tolerances.

**Tech Stack:** Python 3.10+, PyTorch, fairseq2, Unsloth, HuggingFace Transformers

---

## Task 1: Data Extraction Script

**Files:**
- Create: `scripts/extract_fairseq2_batches.py`

**Step 1: Create data extraction script**

```python
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extract batches from fairseq2's data pipeline for convergence testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.datasets import load_dataset
from fairseq2.gang import get_rank, get_world_size, setup_default_gang
from fairseq2.logging import get_log_writer

from recipes.lm.sft.dataset import LMSFTDataset, DataReadOptions, StaticBatching


log = get_log_writer(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract batches from fairseq2 data pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted batches",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to extract",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-3-1b-it",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="hg://facebook/fairseq2-lm-gsm8k",
        help="Dataset path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="sft_train",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup gang (for distributed data loading)
    gang = setup_default_gang()

    # Only rank 0 saves batches
    if get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Extracting {args.num_batches} batches to {args.output_dir}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer, family="hg")
    log.info(f"Loaded tokenizer: {args.tokenizer}")

    # Load dataset
    dataset = load_dataset(
        "lm_sft",
        config={
            "sources": {
                "train": [
                    {
                        "path": args.dataset_path,
                        "split": args.split,
                        "weight": 1.0,
                    }
                ]
            }
        },
    )
    assert isinstance(dataset, LMSFTDataset)
    log.info(f"Loaded dataset: {args.dataset_path}")

    # Create data reader with static batching
    read_options = DataReadOptions(
        batching=StaticBatching(batch_size=1),  # 1 example per GPU
        example_shuffle_window=10_000,
        batch_shuffle_window=0,
        num_accumulate=1,
        prefetch=4,
        source_encode_mode="prompt",
        target_encode_mode="prompt_response",
        chat_mode=False,
        seed=args.seed,
    )

    gangs = type('Gangs', (), {
        'root': gang,
        'dp': gang,
        'tp': type('Gang', (), {'size': 1})(),
        'sdp': type('Gang', (), {'size': 1})(),
    })()

    data_reader = dataset.create_reader(
        split="train",
        tokenizer=tokenizer,
        gangs=gangs,
        min_seq_len=1,
        max_seq_len=args.max_seq_len,
        options=read_options,
    )

    # Extract batches
    log.info("Starting batch extraction...")
    for batch_idx, batch in enumerate(data_reader):
        if batch_idx >= args.num_batches:
            break

        if get_rank() == 0:
            # Extract batch data
            input_batch, target_batch = batch.as_auto_regressive()
            seqs, seqs_layout = input_batch.as_input()

            # Create attention mask from layout
            if seqs_layout.padded:
                attention_mask = (seqs_layout.position_indices >= 0).to(dtype=torch.long)
            else:
                attention_mask = torch.ones_like(seqs, dtype=torch.long)

            # Create labels from targets and target_mask
            labels = target_batch.seqs.clone()
            if target_batch.target_mask is not None:
                labels = labels.masked_fill(~target_batch.target_mask, -100)

            # Save batch
            batch_data = {
                "input_ids": seqs.cpu(),
                "attention_mask": attention_mask.cpu(),
                "labels": labels.cpu(),
                "batch_idx": batch_idx,
            }

            output_path = args.output_dir / f"batch_{batch_idx:04d}.pt"
            torch.save(batch_data, output_path)

            if batch_idx % 10 == 0:
                log.info(f"Extracted batch {batch_idx}/{args.num_batches}")

    if get_rank() == 0:
        log.info(f"Extraction complete! Saved {args.num_batches} batches to {args.output_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the extraction script manually**

```bash
cd /home/richardyue/fairseq2/hg_hardware_test

# Create test output directory
mkdir -p /tmp/test_extraction

# Run extraction (single process for testing)
python3 scripts/extract_fairseq2_batches.py \
    --output-dir /tmp/test_extraction \
    --num-batches 5 \
    --tokenizer google/gemma-3-1b-it

# Verify output
ls -lh /tmp/test_extraction/
# Expected: batch_0000.pt through batch_0004.pt

# Check batch contents
python3 -c "
import torch
batch = torch.load('/tmp/test_extraction/batch_0000.pt')
print('Keys:', batch.keys())
print('input_ids shape:', batch['input_ids'].shape)
print('attention_mask shape:', batch['attention_mask'].shape)
print('labels shape:', batch['labels'].shape)
print('Valid labels count:', (batch['labels'] != -100).sum().item())
"
# Expected: All shapes match, valid labels > 0
```

**Step 3: Commit**

```bash
git add scripts/extract_fairseq2_batches.py
git commit -m "Add fairseq2 batch extraction script for convergence testing

- Extracts batches from fairseq2's SFT data pipeline
- Saves input_ids, attention_mask, labels to disk
- Uses static batching (batch_size=1) for reproducibility
- Only rank 0 saves batches in distributed setting"
```

---

## Task 2: Unsloth Training Script

**Files:**
- Create: `scripts/train_unsloth.py`

**Step 1: Create Unsloth training script**

```python
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train with Unsloth using extracted fairseq2 batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, DistributedSampler

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
    import os
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
```

**Step 2: Test the Unsloth script (dry run)**

```bash
# This will require GPUs and the extracted data from Task 1
# For now, verify syntax is correct
python3 -m py_compile scripts/train_unsloth.py
# Expected: No syntax errors
```

**Step 3: Commit**

```bash
git add scripts/train_unsloth.py
git commit -m "Add Unsloth training script matching fairseq2 hyperparameters

- Uses pre-extracted batches from fairseq2 pipeline
- Matches optimizer: AdamW with lr=3e-4, betas=(0.9,0.95), weight_decay=0.1
- Matches LR schedule: CosineAnnealing with eta_min=6e-5
- Matches training: bf16, grad_clip=1.0, batch_size=1
- Supports distributed training with DDP"
```

---

## Task 3: Checkpoint Comparison Script

**Files:**
- Create: `scripts/compare_checkpoints.py`

**Step 1: Create comparison script**

```python
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare fairseq2 and Unsloth checkpoints for convergence validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_fairseq2_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load fairseq2 checkpoint and extract model state dict."""
    ckpt_path = path / "checkpoint_100.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"fairseq2 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # fairseq2 checkpoints have model under 'model' key
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    return state_dict


def load_unsloth_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load Unsloth checkpoint."""
    ckpt_path = path / "checkpoint_100.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Unsloth checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    return state_dict


def map_parameter_name(fs2_name: str) -> str:
    """
    Map fairseq2 parameter names to HuggingFace/Unsloth names.

    fairseq2 HgCausalLMAdapter wraps HF model under '_wrapped_hf_model',
    so parameters look like:
      _wrapped_hf_model.model.layers.0.self_attn.q_proj.weight

    Unsloth parameters look like:
      model.layers.0.self_attn.q_proj.weight
    """
    if fs2_name.startswith("_wrapped_hf_model."):
        return fs2_name.replace("_wrapped_hf_model.", "")
    return fs2_name


def compare_checkpoints(
    fs2_path: Path,
    unsloth_path: Path,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Compare two checkpoints with tolerance for bfloat16 precision.

    Returns True if all parameters match within tolerance.
    """
    print("Loading checkpoints...")
    fs2_state = load_fairseq2_checkpoint(fs2_path)
    unsloth_state = load_unsloth_checkpoint(unsloth_path)

    print(f"fairseq2 checkpoint: {len(fs2_state)} parameters")
    print(f"Unsloth checkpoint: {len(unsloth_state)} parameters")

    mismatches = []
    matches = []
    missing_in_unsloth = []

    for fs2_key, fs2_param in fs2_state.items():
        unsloth_key = map_parameter_name(fs2_key)

        if unsloth_key not in unsloth_state:
            missing_in_unsloth.append(unsloth_key)
            continue

        unsloth_param = unsloth_state[unsloth_key]

        # Check shape match
        if fs2_param.shape != unsloth_param.shape:
            print(f"Shape mismatch: {unsloth_key}")
            print(f"  fairseq2: {fs2_param.shape}")
            print(f"  Unsloth:  {unsloth_param.shape}")
            continue

        # Convert to float32 for comparison
        fs2_val = fs2_param.float().cpu().numpy()
        unsloth_val = unsloth_param.float().cpu().numpy()

        # Check if values match within tolerance
        if np.allclose(fs2_val, unsloth_val, rtol=rtol, atol=atol):
            matches.append(unsloth_key)
        else:
            max_diff = np.abs(fs2_val - unsloth_val).max()
            mean_diff = np.abs(fs2_val - unsloth_val).mean()
            rel_diff = max_diff / (np.abs(fs2_val).max() + 1e-8)

            mismatches.append({
                "param": unsloth_key,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "rel_diff": rel_diff,
            })

    # Print report
    print(f"\n{'='*70}")
    print(f"Convergence Report")
    print(f"{'='*70}")
    print(f"Total parameters:      {len(fs2_state)}")
    print(f"Matching parameters:   {len(matches)}")
    print(f"Mismatching parameters: {len(mismatches)}")
    print(f"Missing in Unsloth:    {len(missing_in_unsloth)}")
    print(f"\nTolerance: rtol={rtol}, atol={atol}")

    if missing_in_unsloth:
        print(f"\n{'='*70}")
        print("Missing in Unsloth (first 10):")
        print(f"{'='*70}")
        for key in missing_in_unsloth[:10]:
            print(f"  {key}")

    if mismatches:
        print(f"\n{'='*70}")
        print("Mismatches (top 10 by max absolute difference):")
        print(f"{'='*70}")
        sorted_mismatches = sorted(mismatches, key=lambda x: x["max_diff"], reverse=True)
        for m in sorted_mismatches[:10]:
            print(f"\n{m['param']}")
            print(f"  Max diff:  {m['max_diff']:.6e}")
            print(f"  Mean diff: {m['mean_diff']:.6e}")
            print(f"  Rel diff:  {m['rel_diff']:.6e}")

    converged = len(mismatches) == 0 and len(missing_in_unsloth) == 0

    if converged:
        print(f"\n{'='*70}")
        print("✓ SUCCESS: All parameters match within tolerance!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("✗ FAILURE: Checkpoints diverged")
        print(f"{'='*70}")

    return converged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fairseq2 and Unsloth checkpoints"
    )
    parser.add_argument(
        "--fs2-checkpoint",
        type=Path,
        required=True,
        help="Path to fairseq2 checkpoint directory",
    )
    parser.add_argument(
        "--unsloth-checkpoint",
        type=Path,
        required=True,
        help="Path to Unsloth checkpoint directory",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for np.allclose",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for np.allclose",
    )

    args = parser.parse_args()

    converged = compare_checkpoints(
        args.fs2_checkpoint,
        args.unsloth_checkpoint,
        rtol=args.rtol,
        atol=args.atol,
    )

    exit(0 if converged else 1)


if __name__ == "__main__":
    main()
```

**Step 2: Test comparison script (syntax check)**

```bash
python3 -m py_compile scripts/compare_checkpoints.py
# Expected: No syntax errors
```

**Step 3: Commit**

```bash
git add scripts/compare_checkpoints.py
git commit -m "Add checkpoint comparison script for convergence validation

- Loads fairseq2 and Unsloth checkpoints
- Maps parameter names (_wrapped_hf_model prefix handling)
- Uses np.allclose with bf16-appropriate tolerances
- Reports detailed mismatch information (max/mean/rel diff)
- Returns exit code 0 for success, 1 for failure"
```

---

## Task 4: Bash Orchestration Script

**Files:**
- Create: `scripts/run_comparison.sh`

**Step 1: Create orchestration script**

```bash
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e  # Exit on error

echo "=========================================="
echo "fairseq2 vs Unsloth Convergence Test"
echo "=========================================="

# Configuration
FS2_OUT="/checkpoint/seamless/richardyue/fs2_out"
UNSLOTH_OUT="/checkpoint/seamless/richardyue/unsloth_out"
EXTRACTED_DATA="/checkpoint/seamless/richardyue/extracted_data"
NUM_STEPS=100

# Parse command line arguments
SKIP_EXTRACTION=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Setup
echo -e "\n[1/4] Setting up directories..."
mkdir -p "$FS2_OUT" "$UNSLOTH_OUT" "$EXTRACTED_DATA"

# Step 2: Extract dataset
if [ "$SKIP_EXTRACTION" = false ]; then
    echo -e "\n[2/4] Extracting dataset from fairseq2..."
    python3 scripts/extract_fairseq2_batches.py \
        --output-dir "$EXTRACTED_DATA" \
        --num-batches $NUM_STEPS \
        --tokenizer google/gemma-3-1b-it \
        --dataset-path hg://facebook/fairseq2-lm-gsm8k \
        --split sft_train \
        --max-seq-len 4096 \
        --seed 2

    echo "Dataset extraction complete!"
else
    echo -e "\n[2/4] Skipping dataset extraction (--skip-extraction flag set)"

    # Verify extracted data exists
    if [ ! -f "$EXTRACTED_DATA/batch_0000.pt" ]; then
        echo "ERROR: Extracted data not found at $EXTRACTED_DATA"
        echo "Run without --skip-extraction first"
        exit 1
    fi
fi

# Step 3: Run parallel training
echo -e "\n[3/4] Running parallel training on 2x4 GPUs..."
echo "  - fairseq2 on GPUs 0-3"
echo "  - Unsloth on GPUs 4-7"

# fairseq2 on GPUs 0-3
echo "Starting fairseq2 training..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
    -m recipes.lm.sft \
    --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
    "$FS2_OUT" &
FS2_PID=$!
echo "fairseq2 training started (PID: $FS2_PID)"

# Unsloth on GPUs 4-7
echo "Starting Unsloth training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
    scripts/train_unsloth.py \
    --data-dir "$EXTRACTED_DATA" \
    --output-dir "$UNSLOTH_OUT" \
    --num-steps $NUM_STEPS \
    --model-name google/gemma-3-1b-it &
UNSLOTH_PID=$!
echo "Unsloth training started (PID: $UNSLOTH_PID)"

# Wait for both to complete
echo -e "\nWaiting for training to complete..."
echo "fairseq2 PID: $FS2_PID"
echo "Unsloth PID: $UNSLOTH_PID"

wait $FS2_PID
FS2_EXIT=$?

wait $UNSLOTH_PID
UNSLOTH_EXIT=$?

if [ $FS2_EXIT -ne 0 ]; then
    echo "ERROR: fairseq2 training failed with exit code $FS2_EXIT"
    exit 1
fi

if [ $UNSLOTH_EXIT -ne 0 ]; then
    echo "ERROR: Unsloth training failed with exit code $UNSLOTH_EXIT"
    exit 1
fi

echo "Both training runs completed successfully!"

# Step 4: Compare checkpoints
echo -e "\n[4/4] Comparing checkpoints..."
python3 scripts/compare_checkpoints.py \
    --fs2-checkpoint "$FS2_OUT" \
    --unsloth-checkpoint "$UNSLOTH_OUT" \
    --rtol 1e-3 \
    --atol 1e-5

COMPARE_EXIT=$?

if [ $COMPARE_EXIT -eq 0 ]; then
    echo -e "\n=========================================="
    echo "✓ SUCCESS: Checkpoints match!"
    echo "=========================================="
else
    echo -e "\n=========================================="
    echo "✗ FAILURE: Checkpoints diverged"
    echo "=========================================="
fi

exit $COMPARE_EXIT
```

**Step 2: Make script executable**

```bash
chmod +x scripts/run_comparison.sh
```

**Step 3: Commit**

```bash
git add scripts/run_comparison.sh
git commit -m "Add orchestration script for parallel convergence test

- Sets up directories for fairseq2, Unsloth, extracted data
- Extracts dataset batches (or skips with --skip-extraction)
- Runs fairseq2 and Unsloth training in parallel on separate GPUs
- Compares checkpoints and reports results
- Returns exit code 0 for success, 1 for failure"
```

---

## Task 5: Update Config for 100-Step Test

**Files:**
- Modify: `recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml`

**Step 1: Update regime section for 100-step test**

Change the `regime` section:

```yaml
regime:
  num_steps: 100  # Changed from 100000
  checkpoint_every_n_steps: 100  # Checkpoint at end
  validate_every_n_steps: 1000  # No validation during test
  checkpoint_every_n_data_epochs: 100
  keep_last_n_checkpoints: 1  # Only keep final checkpoint
  publish_metrics_every_n_steps: 10
  save_model_only: false
  export_hugging_face: false  # Disable HF export for HG models
```

**Step 2: Verify the change**

```bash
grep -A 8 "^regime:" recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml
# Expected: Shows num_steps: 100 and keep_last_n_checkpoints: 1
```

**Step 3: Commit**

```bash
git add recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml
git commit -m "Configure gemma_3_1b config for 100-step convergence test

- Set num_steps to 100 for quick convergence test
- Checkpoint only at step 100
- Keep only last checkpoint to save space"
```

---

## Task 6: Add README Documentation

**Files:**
- Create: `scripts/README.md`

**Step 1: Create README for scripts**

```markdown
# Convergence Testing Scripts

Scripts for validating that fairseq2's HuggingFace model adapter produces identical training convergence to Unsloth.

## Overview

The test extracts batches from fairseq2's data pipeline, runs parallel training on separate GPUs (fairseq2 and Unsloth), then compares the resulting checkpoints.

## Quick Start

```bash
# Run the full test (requires 8 GPUs)
./scripts/run_comparison.sh
```

## Requirements

- 8 GPUs (2x4 for parallel training)
- fairseq2 installed with HuggingFace model support
- Unsloth installed
- Access to `/checkpoint/seamless/richardyue/` for outputs

## Scripts

### `run_comparison.sh`

Main orchestration script that runs the entire test pipeline.

**Usage:**
```bash
# Full test (extraction + training + comparison)
./scripts/run_comparison.sh

# Skip extraction (reuse previously extracted data)
./scripts/run_comparison.sh --skip-extraction
```

**Exit codes:**
- `0`: Success - checkpoints match
- `1`: Failure - checkpoints diverged or training failed

### `extract_fairseq2_batches.py`

Extracts batches from fairseq2's data pipeline to disk.

**Usage:**
```bash
python3 scripts/extract_fairseq2_batches.py \
    --output-dir /path/to/output \
    --num-batches 100 \
    --tokenizer google/gemma-3-1b-it
```

**Options:**
- `--output-dir`: Directory to save extracted batches
- `--num-batches`: Number of batches to extract (default: 100)
- `--tokenizer`: Tokenizer to use (default: google/gemma-3-1b-it)
- `--dataset-path`: Dataset path (default: hg://facebook/fairseq2-lm-gsm8k)
- `--split`: Dataset split (default: sft_train)
- `--max-seq-len`: Maximum sequence length (default: 4096)
- `--seed`: Random seed (default: 2)

### `train_unsloth.py`

Trains with Unsloth using extracted batches and matched hyperparameters.

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
    scripts/train_unsloth.py \
    --data-dir /path/to/extracted_data \
    --output-dir /path/to/output \
    --num-steps 100
```

**Hyperparameters (matched to fairseq2):**
- Optimizer: AdamW with lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
- LR Scheduler: CosineAnnealingLR with T_max=100, eta_min=6e-5
- Precision: bfloat16
- Gradient clipping: max_norm=1.0
- Batch size: 1 per GPU

### `compare_checkpoints.py`

Compares fairseq2 and Unsloth checkpoints.

**Usage:**
```bash
python3 scripts/compare_checkpoints.py \
    --fs2-checkpoint /path/to/fs2_out \
    --unsloth-checkpoint /path/to/unsloth_out \
    --rtol 1e-3 \
    --atol 1e-5
```

**Options:**
- `--rtol`: Relative tolerance for np.allclose (default: 1e-3)
- `--atol`: Absolute tolerance for np.allclose (default: 1e-5)

**Output:**
- Prints convergence report with match/mismatch counts
- Shows top 10 mismatches by absolute difference
- Returns exit code 0 if all parameters match, 1 otherwise

## Architecture

1. **Data extraction**: fairseq2's data pipeline → batches on disk
2. **Parallel training**: fairseq2 (GPUs 0-3) + Unsloth (GPUs 4-7)
3. **Checkpoint comparison**: np.allclose with bf16 tolerances

## Output Directories

- `/checkpoint/seamless/richardyue/fs2_out/`: fairseq2 checkpoint
- `/checkpoint/seamless/richardyue/unsloth_out/`: Unsloth checkpoint
- `/checkpoint/seamless/richardyue/extracted_data/`: Extracted batches

## Troubleshooting

**Training fails with OOM:**
- Reduce `--max-seq-len` in extraction script
- Check GPU memory with `nvidia-smi`

**Checkpoints diverge:**
- Check hyperparameters match exactly
- Verify extracted data is identical for both runs
- Increase tolerances (--rtol, --atol) if differences are small

**Data extraction fails:**
- Verify dataset path is accessible
- Check tokenizer can be loaded
- Ensure sufficient disk space in output directory
```

**Step 2: Commit**

```bash
git add scripts/README.md
git commit -m "Add documentation for convergence testing scripts

- Usage instructions for each script
- Architecture overview
- Troubleshooting guide"
```

---

## Summary

After completing all tasks, you will have:

1. ✅ **Data extraction script** - Extracts batches from fairseq2 pipeline
2. ✅ **Unsloth training script** - Trains with matched hyperparameters
3. ✅ **Comparison script** - Validates convergence with tolerances
4. ✅ **Orchestration script** - Runs full test pipeline
5. ✅ **Updated config** - Set for 100-step test
6. ✅ **Documentation** - Complete usage guide

**To run the test:**

```bash
cd /home/richardyue/fairseq2/hg_hardware_test
./scripts/run_comparison.sh
```

**Expected outcome:**
- Both training runs complete successfully
- Checkpoint comparison shows 0 mismatches
- Exit code 0 (success)

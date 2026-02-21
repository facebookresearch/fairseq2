# Convergence Testing Scripts

This directory contains scripts for validating that fairseq2's HuggingFace adapter produces identical convergence to Unsloth. The test compares model weights after training on identical batches with matched hyperparameters.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Convergence Test Flow                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. Extract fairseq2 Dataset Batches  │
        │     (extract_fairseq2_batches.py)     │
        └───────────────────────────────────────┘
                            │
                            ▼
                  out/extracted_batches/
                  ├── batch_0000.pt
                  ├── batch_0001.pt
                  └── ...
                            │
           ┌────────────────┴────────────────┐
           │                                 │
           ▼                                 ▼
┌────────────────────┐           ┌────────────────────┐
│  2a. fairseq2 SFT  │           │  2b. Unsloth Train │
│  (fairseq2 lm cmd) │           │ (train_unsloth.py) │
│                    │           │                    │
│  GPUs 0-3          │           │  GPUs 4-7          │
└────────────────────┘           └────────────────────┘
           │                                 │
           ▼                                 ▼
out/fairseq2_checkpoints/      out/unsloth_checkpoints/
└── checkpoint_100.pt          └── checkpoint_100.pt
           │                                 │
           └────────────────┬────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  3. Compare Checkpoints  │
              │ (compare_checkpoints.py) │
              └──────────────────────────┘
                            │
                            ▼
                  ✓ PASS or ✗ FAIL
```

## Quick Start

### Full Test (Automated)
```bash
# Run complete convergence test (extraction + parallel training + comparison)
./scripts/run_comparison.sh

# Skip extraction if batches already exist
./scripts/run_comparison.sh --skip-extraction
```

### Manual Step-by-Step

```bash
# 1. Extract batches
python scripts/extract_fairseq2_batches.py \
    --output-dir out/extracted_batches \
    --num-batches 100

# 2a. Train with fairseq2
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq2 lm gemma3_2b_it \
    --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
    --output-dir out/fairseq2_checkpoints

# 2b. Train with Unsloth (in parallel)
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/train_unsloth.py \
    --data-dir out/extracted_batches \
    --output-dir out/unsloth_checkpoints \
    --num-steps 100

# 3. Compare checkpoints
python scripts/compare_checkpoints.py \
    --fairseq2-checkpoint out/fairseq2_checkpoints/checkpoint_100.pt \
    --unsloth-checkpoint out/unsloth_checkpoints/checkpoint_100.pt
```

## Script Descriptions

### 1. `extract_fairseq2_batches.py`

Extracts batches from fairseq2's data pipeline for reproducible training.

**Purpose:**
- Ensures both frameworks train on identical data
- Eliminates data loading as a source of variance
- Creates reproducible test conditions

**Usage:**
```bash
python scripts/extract_fairseq2_batches.py \
    --output-dir out/extracted_batches \
    --num-batches 100 \
    --tokenizer google/gemma-3-2b-it \
    --dataset-path hg://facebook/fairseq2-lm-gsm8k \
    --split sft_train \
    --max-seq-len 4096 \
    --seed 2
```

**Arguments:**
- `--output-dir`: Directory to save extracted batches (required)
- `--num-batches`: Number of batches to extract (default: 100)
- `--tokenizer`: Tokenizer to use (default: google/gemma-3-2b-it)
- `--dataset-path`: Dataset path (default: hg://facebook/fairseq2-lm-gsm8k)
- `--split`: Dataset split (default: sft_train)
- `--max-seq-len`: Maximum sequence length (default: 4096)
- `--seed`: Random seed (default: 2)

**Output Format:**
Each batch is saved as `batch_XXXX.pt` containing:
```python
{
    "input_ids": Tensor,       # Shape: [1, seq_len]
    "attention_mask": Tensor,  # Shape: [1, seq_len]
    "labels": Tensor,          # Shape: [1, seq_len], -100 for non-targets
    "batch_idx": int,
}
```

### 2. `train_unsloth.py`

Trains using Unsloth with extracted batches, matching fairseq2 hyperparameters exactly.

**Purpose:**
- Provides baseline convergence using Unsloth
- Uses identical batches and hyperparameters as fairseq2
- Supports distributed training for fair comparison

**Usage:**
```bash
# Single GPU
python scripts/train_unsloth.py \
    --data-dir out/extracted_batches \
    --output-dir out/unsloth_checkpoints \
    --num-steps 100

# Multi-GPU (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    scripts/train_unsloth.py \
    --data-dir out/extracted_batches \
    --output-dir out/unsloth_checkpoints \
    --num-steps 100
```

**Arguments:**
- `--data-dir`: Directory with extracted batches (required)
- `--output-dir`: Directory to save checkpoint (required)
- `--num-steps`: Number of training steps (default: 100)
- `--model-name`: Model name (default: google/gemma-3-1b-it)

**Output:**
- `checkpoint_100.pt`: Model checkpoint at step 100

### 3. `compare_checkpoints.py`

Compares fairseq2 and Unsloth checkpoints to validate convergence.

**Purpose:**
- Validates numerical equivalence of trained weights
- Reports detailed mismatch statistics
- Provides pass/fail decision for convergence test

**Usage:**
```bash
python scripts/compare_checkpoints.py \
    --fairseq2-checkpoint out/fairseq2_checkpoints/checkpoint_100.pt \
    --unsloth-checkpoint out/unsloth_checkpoints/checkpoint_100.pt \
    --rtol 1e-3 \
    --atol 1e-5
```

**Arguments:**
- `--fairseq2-checkpoint`: Path to fairseq2 checkpoint (required)
- `--unsloth-checkpoint`: Path to Unsloth checkpoint (required)
- `--rtol`: Relative tolerance for np.allclose (default: 1e-3)
- `--atol`: Absolute tolerance for np.allclose (default: 1e-5)
- `--top-n`: Number of worst mismatches to report (default: 10)

**Exit Codes:**
- `0`: All parameters match within tolerance (PASS)
- `1`: Some parameters differ beyond tolerance (FAIL)

**Output:**
```
================================================================================
Checkpoint Comparison for Convergence Validation
================================================================================
fairseq2 checkpoint: out/fairseq2_checkpoints/checkpoint_100.pt
Unsloth checkpoint:  out/unsloth_checkpoints/checkpoint_100.pt
Tolerances: rtol=0.001, atol=1e-05
================================================================================

Comparing 266 common parameters...

================================================================================
✓ All parameters match within tolerance!
================================================================================

✓ SUCCESS: All parameters match within tolerance
```

### 4. `run_comparison.sh`

Orchestrates the complete convergence test workflow.

**Purpose:**
- Automates end-to-end testing
- Manages parallel training on separate GPUs
- Provides clear pass/fail results

**Usage:**
```bash
# Full test
./scripts/run_comparison.sh

# Skip extraction (use existing batches)
./scripts/run_comparison.sh --skip-extraction
```

**Flags:**
- `--skip-extraction`: Skip batch extraction step (requires existing batches)

**Output Directories:**
- `out/extracted_batches/`: Extracted training batches
- `out/fairseq2_checkpoints/`: fairseq2 checkpoints and logs
- `out/unsloth_checkpoints/`: Unsloth checkpoints and logs

## Hyperparameter Specifications

All hyperparameters are matched between fairseq2 and Unsloth:

### Model Configuration
```yaml
Model: google/gemma-3-2b-it
Architecture: causal_lm (decoder-only)
Precision: bfloat16
Quantization: None (full precision for accurate comparison)
Max Sequence Length: 4096
```

### Dataset Configuration
```yaml
Dataset: hg://facebook/fairseq2-lm-gsm8k
Split: sft_train
Batch Size: 1 per GPU (4 total with 4 GPUs)
Shuffle Window: 10,000 examples
Seed: 2
Chat Mode: false (standard SFT)
```

### Training Configuration
```yaml
Optimizer: AdamW (fused)
Learning Rate: 3e-4
Betas: (0.9, 0.95)
Epsilon: 1e-8
Weight Decay: 0.1
LR Schedule: Cosine Annealing
  - T_max: 100 steps
  - eta_min: 6e-5 (0.2 × base_lr)
Gradient Clipping: max_norm=1.0
Training Steps: 100
```

### Checkpoint Configuration
```yaml
Checkpoint Frequency: Every 100 steps
Checkpoints Saved: checkpoint_100.pt
Validation: After training completes
```

## Output Directories

### `out/extracted_batches/`
Contains pre-extracted training batches:
```
out/extracted_batches/
├── batch_0000.pt  # {input_ids, attention_mask, labels}
├── batch_0001.pt
├── ...
└── batch_0099.pt
```

### `out/fairseq2_checkpoints/`
Contains fairseq2 training outputs:
```
out/fairseq2_checkpoints/
├── checkpoint_100.pt  # Model checkpoint
├── training.log       # Training logs
└── metrics/           # Training metrics
```

### `out/unsloth_checkpoints/`
Contains Unsloth training outputs:
```
out/unsloth_checkpoints/
├── checkpoint_100.pt  # Model checkpoint
└── training.log       # Training logs
```

## Troubleshooting

### Extraction Issues

**Problem:** `ModuleNotFoundError: No module named 'fairseq2'`
```bash
# Solution: Install fairseq2 in development mode
pip install -e .
```

**Problem:** `Dataset not found: hg://facebook/fairseq2-lm-gsm8k`
```bash
# Solution: Dataset will be downloaded on first run
# Ensure internet connection is available
```

**Problem:** `CUDA out of memory during extraction`
```bash
# Solution: Reduce batch size or max sequence length
python scripts/extract_fairseq2_batches.py \
    --max-seq-len 2048 \  # Reduce from 4096
    --output-dir out/extracted_batches
```

### Training Issues

**Problem:** `RuntimeError: NCCL error during distributed training`
```bash
# Solution: Check GPU availability and CUDA_VISIBLE_DEVICES
nvidia-smi  # Verify GPUs are available

# Ensure non-overlapping GPU sets
CUDA_VISIBLE_DEVICES=0,1,2,3  # fairseq2
CUDA_VISIBLE_DEVICES=4,5,6,7  # Unsloth
```

**Problem:** `fairseq2 training hangs during initialization`
```bash
# Solution: Check for distributed environment variables
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
# Then restart fairseq2 training
```

**Problem:** `Unsloth: File not found error for batch files`
```bash
# Solution: Ensure extraction completed successfully
ls -l out/extracted_batches/
# Should contain batch_0000.pt through batch_0099.pt
```

**Problem:** `Learning rate differs between frameworks`
```bash
# Solution: Verify scheduler configuration
# Check training logs for LR values:
grep "LR:" out/fairseq2_checkpoints/training.log
grep "LR:" out/unsloth_checkpoints/training.log
```

### Comparison Issues

**Problem:** `Checkpoint file not found`
```bash
# Solution: Verify checkpoints were saved
ls out/fairseq2_checkpoints/checkpoint_100.pt
ls out/unsloth_checkpoints/checkpoint_100.pt
```

**Problem:** `Shape mismatch between checkpoints`
```bash
# Solution: This indicates a model architecture mismatch
# Verify both use the same model: google/gemma-3-2b-it
# Check fairseq2 config: recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml
# Check Unsloth args: --model-name google/gemma-3-2b-it
```

**Problem:** `Value mismatch beyond tolerance`
```bash
# Solution: Check for numerical precision issues
# 1. Verify both use bfloat16
# 2. Increase tolerance if using lower precision:
python scripts/compare_checkpoints.py \
    --fairseq2-checkpoint out/fairseq2_checkpoints/checkpoint_100.pt \
    --unsloth-checkpoint out/unsloth_checkpoints/checkpoint_100.pt \
    --rtol 1e-2  # Increase from 1e-3
```

**Problem:** `Parameters only in fairseq2: _wrapped_hf_model.*`
```bash
# Solution: This is expected - the script automatically maps names
# fairseq2 wraps HF models with "_wrapped_hf_model." prefix
# The comparison script strips this prefix automatically
```

### Resource Issues

**Problem:** `Not enough GPUs for parallel training`
```bash
# Solution: Run sequentially instead
# 1. Train fairseq2 first
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq2 lm gemma3_2b_it \
    --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
    --output-dir out/fairseq2_checkpoints

# 2. Then train Unsloth
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_unsloth.py \
    --data-dir out/extracted_batches \
    --output-dir out/unsloth_checkpoints

# 3. Compare
python scripts/compare_checkpoints.py \
    --fairseq2-checkpoint out/fairseq2_checkpoints/checkpoint_100.pt \
    --unsloth-checkpoint out/unsloth_checkpoints/checkpoint_100.pt
```

**Problem:** `Disk space full during training`
```bash
# Solution: Clean up old checkpoints
rm -rf out/fairseq2_checkpoints/checkpoint_*.pt
rm -rf out/unsloth_checkpoints/checkpoint_*.pt
# Keep only checkpoint_100.pt for comparison
```

## Validation Criteria

The convergence test **PASSES** if:
1. Both training runs complete without errors
2. Checkpoints are created at step 100
3. All common parameters match within tolerance:
   - Relative tolerance: 1e-3 (0.1%)
   - Absolute tolerance: 1e-5
4. No shape mismatches between parameters

The test **FAILS** if:
- Any training run crashes
- Checkpoints are missing
- Parameter values differ beyond tolerance
- Shape mismatches occur

## Expected Results

For a successful test with 100 training steps:

```
=== SUCCESS: Convergence test passed! ===
Checkpoints are identical within tolerance.

Statistics:
- Total parameters compared: 266
- Parameters matched: 266 (100%)
- Max absolute difference: 2.3e-6
- Mean absolute difference: 4.1e-7
- Max relative difference: 0.08%
```

## Development Notes

### Adding New Tests

To test with different configurations:

1. Create new config file in `recipes/lm/sft/configs/`
2. Update hyperparameters in `train_unsloth.py` to match
3. Update `run_comparison.sh` with new config path

### Debugging Convergence Issues

If tests fail, compare training logs:

```bash
# Check training losses
grep "Loss:" out/fairseq2_checkpoints/training.log | head -20
grep "Loss:" out/unsloth_checkpoints/training.log | head -20

# Check learning rates
grep "LR:" out/fairseq2_checkpoints/training.log | head -20
grep "LR:" out/unsloth_checkpoints/training.log | head -20

# Compare gradients (add logging to scripts)
# Compare optimizer states (modify checkpoint saving)
```

### Performance Benchmarking

To measure training speed:

```bash
# Time fairseq2
time CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq2 lm gemma3_2b_it ...

# Time Unsloth
time CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/train_unsloth.py ...
```

## References

- fairseq2 Documentation: https://github.com/facebookresearch/fairseq2
- Unsloth Documentation: https://github.com/unslothai/unsloth
- GSM8K Dataset: https://github.com/openai/grade-school-math
- Gemma Model Card: https://huggingface.co/google/gemma-3-2b-it

## License

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

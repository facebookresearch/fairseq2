# fairseq2 vs Unsloth Convergence Test Design

**Date:** 2026-02-18
**Purpose:** Validate that fairseq2's HuggingFace model adapter produces identical training convergence to Unsloth when given identical data and hyperparameters.

## Overview

This system runs parallel training of the same model (google/gemma-3-1b-it) on identical data using fairseq2 and Unsloth, then compares the resulting checkpoints to verify convergence. The test runs for 100 training steps on 2x4 GPUs (embarrassingly parallel).

## Architecture

### Components

1. **Modified fairseq2 config** - Uses existing config with 100 step limit
2. **Data extraction script** (`scripts/extract_fairseq2_batches.py`) - Extracts processed batches from fairseq2's data pipeline
3. **Unsloth training script** (`scripts/train_unsloth.py`) - Trains using extracted batches with matched hyperparameters
4. **Comparison script** (`scripts/compare_checkpoints.py`) - Compares checkpoints with appropriate tolerances
5. **Orchestration script** (`scripts/run_comparison.sh`) - Coordinates the entire workflow

### Workflow

```
1. Setup directories
2. Extract dataset (fairseq2 pipeline → disk)
3. Parallel training:
   - fairseq2 on GPUs 0-3
   - Unsloth on GPUs 4-7
4. Compare checkpoints (np.allclose with bf16 tolerances)
```

## Data Extraction Strategy

Extract batches by running fairseq2's data pipeline for 100 steps and saving to disk:

```python
# scripts/extract_fairseq2_batches.py
for i, batch in enumerate(data_reader):
    if i >= 100:
        break
    torch.save({
        'input_ids': batch.seqs,
        'attention_mask': ...,  # derived from batch layout
        'labels': ...,          # derived from target_mask
        'batch_idx': i
    }, f'{output_dir}/batch_{i:04d}.pt')
```

This guarantees both frameworks train on identical data in identical order.

## Hyperparameter Matching

All critical training parameters must match exactly between fairseq2 and Unsloth:

### Optimizer (AdamW)
- Learning rate: `3e-4`
- Betas: `(0.9, 0.95)`
- Epsilon: `1e-8`
- Weight decay: `0.1`
- Implementation: Fused AdamW

### LR Scheduler (Cosine Annealing)
- Base LR: `3e-4`
- Final LR: `6e-5` (base * 0.2)
- Warmup steps: `0`
- Cycle length: `100` steps
- Formula: `eta_t = 6e-5 + 0.5 * (3e-4 - 6e-5) * (1 + cos(pi * t / 100))`

### Training Configuration
- Mixed precision: bfloat16 (static mode)
- Gradient accumulation: 1 batch
- Max gradient norm: 1.0
- Batch size: 1 per GPU (4 total across 4 GPUs)
- Training steps: 100

## Implementation Details

### Unsloth Training Script

```python
# scripts/train_unsloth.py
import torch
from unsloth import FastLanguageModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load model (no quantization for fair comparison)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Optimizer matching fairseq2 exactly
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,
)

# LR Scheduler matching fairseq2's cosine annealing
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=6e-5,
)

# Training loop with gradient clipping
for step, batch in enumerate(dataloader):
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Checkpoint Comparison

```python
# scripts/compare_checkpoints.py
def compare_checkpoints(fs2_path, unsloth_path, rtol=1e-3, atol=1e-5):
    """
    Compare checkpoints with tolerances appropriate for bfloat16.

    Maps parameter names:
    - fairseq2: _wrapped_hf_model.model.layers.0.self_attn.q_proj.weight
    - Unsloth:  model.layers.0.self_attn.q_proj.weight
    """
    # Load both checkpoints
    # Map parameter names (strip _wrapped_hf_model prefix)
    # Use np.allclose() with bf16-appropriate tolerances
    # Report mismatches with max/mean differences
```

### Bash Orchestration

```bash
# scripts/run_comparison.sh
#!/bin/bash
set -e

# 1. Setup
mkdir -p {fs2_out,unsloth_out,extracted_data}

# 2. Extract dataset
python3 scripts/extract_fairseq2_batches.py \
    --num-batches 100

# 3. Parallel training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
    -m recipes.lm.sft \
    --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
    /checkpoint/seamless/richardyue/fs2_out &

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
    scripts/train_unsloth.py \
    --output-dir /checkpoint/seamless/richardyue/unsloth_out &

wait

# 4. Compare
python3 scripts/compare_checkpoints.py
```

## Success Criteria

The test passes if:
- Both training runs complete without errors
- All model parameters match within `rtol=1e-3, atol=1e-5`
- Convergence report shows 0 mismatches

## Error Reporting

If convergence fails, the comparison script prints:
- Total parameters vs matching parameters
- Top 10 worst mismatches by maximum difference
- For each mismatch: parameter name, max diff, mean diff

## File Structure

```
scripts/
├── extract_fairseq2_batches.py    # Extract data from fs2 pipeline
├── train_unsloth.py                # Unsloth training with matched params
├── compare_checkpoints.py          # Checkpoint comparison utility
└── run_comparison.sh               # Main orchestration script

/checkpoint/seamless/richardyue/
├── fs2_out/                        # fairseq2 checkpoint
├── unsloth_out/                    # Unsloth checkpoint
└── extracted_data/                 # Extracted batches
    ├── batch_0000.pt
    ├── batch_0001.pt
    └── ...
```

## Notes

- Uses embarrassingly parallel execution (independent 4-GPU jobs)
- Data extraction ensures identical training data and order
- Tolerances account for bfloat16 precision differences
- Exit codes enable CI integration

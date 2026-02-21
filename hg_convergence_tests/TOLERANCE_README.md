# Convergence Testing: fairseq2 HG Adapter vs HuggingFace

This document describes the requirements for achieving exact convergence when comparing fairseq2's HuggingFace adapter against native HuggingFace Transformers training.

## Key Findings

### Root Cause: Non-Deterministic Flash Attention

The primary source of gradient divergence was **Flash Attention**, which uses non-deterministic algorithms by default. Even with identical models, data, and PyTorch determinism settings, Flash Attention causes different gradients on each backward pass.

**Symptom**: Models start with identical loss (7.4044) but diverge after the first gradient update.

**Solution**: Disable Flash Attention by setting `attn_implementation="eager"` when loading models.

## Requirements for Exact Convergence

### 1. Disable Flash Attention

Both training scripts must disable Flash Attention:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager",  # Critical for determinism
)
```

Or via fairseq2 HG adapter:

```python
model = load_hg_model_simple(
    model_name,
    model_type="causal_lm",
    trust_remote_code=True,
    dtype="bfloat16",
    attn_implementation="eager",  # Critical for determinism
)
```

### 2. Enable Full PyTorch Determinism

Set these **before any CUDA operations**:

```python
import os
import torch

# Environment variables
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# PyTorch determinism settings
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 3. Disable Dropout

Use `model.eval()` mode to disable dropout while still allowing gradient updates:

```python
model.eval()  # Disables dropout and batch norm updates
# Gradients still flow, weights still update
```

### 4. Identical Data Ordering

Both frameworks must process **exactly the same batches in the same order**:

- Extract batches once using `scripts/extract_fairseq2_batches.py`
- Both training scripts load from the same extracted batch files
- No shuffling, no distributed sampling

### 5. Identical Training Configuration

Match all hyperparameters:

```python
# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,
)

# LR Scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=500,
    eta_min=6e-5,
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Random seed
seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### 6. Run on Same GPU

Run both trainings on the **same GPU** to avoid GPU-specific numerical differences:

```bash
CUDA_VISIBLE_DEVICES=0 python train_fairseq2_from_batches.py ...
CUDA_VISIBLE_DEVICES=0 python train_hf_single_gpu.py ...
```

Run sequentially (not in parallel) to avoid resource contention.

## Verification

### Expected Results

With all requirements met:

1. **Initial forward pass**: Identical loss (difference = 0.0)
2. **Gradients**: Identical (max diff < 1e-8)
3. **Training trajectory**: Identical step-by-step losses
4. **Final checkpoints**: All parameters match within tolerance

### Comparison Tolerances

For bfloat16 training with deterministic settings:

```python
rtol = 1e-5  # Relative tolerance
atol = 1e-7  # Absolute tolerance
```

Use stricter tolerances when full determinism is enabled.

### Diagnostic Commands

Check for Flash Attention warnings:

```bash
grep -i "flash attention" training.log
```

Expected: No warnings about non-deterministic Flash Attention.

Compare training curves:

```bash
python scripts/compare_training_curves.py
```

Expected: Step-by-step losses should be identical.

Compare final checkpoints:

```bash
python scripts/compare_checkpoints_simple.py \
    --fairseq2-checkpoint fairseq2_checkpoints/checkpoint_500.pt \
    --hf-checkpoint hf_checkpoints/checkpoint_500.pt
```

Expected: All 341 parameters match within tolerance.

## Troubleshooting

### Issue: Gradients differ even with determinism enabled

**Check**: Flash Attention warning in logs

```bash
grep "Flash Attention defaults to a non-deterministic algorithm" training.log
```

**Fix**: Add `attn_implementation="eager"` to model loading

### Issue: Forward pass identical but gradients differ

**Cause**: Non-deterministic backward operations (Flash Attention, CUBLAS)

**Fix**: Ensure all determinism settings are enabled **before** model loading

### Issue: Training curves diverge gradually

**Cause**: Dropout enabled or different data ordering

**Fix**:
- Verify `model.eval()` is called
- Verify both scripts load from same batch files

### Issue: Different initial loss

**Cause**: Models loaded with different weights or different data

**Fix**:
- Check both models load from same checkpoint
- Verify batch_0000.pt is identical for both runs

## Performance Impact

**Flash Attention disabled**: ~2-3x slower training but fully deterministic

For production training where exact reproducibility isn't required, Flash Attention should be enabled for performance.

## File Structure

```
scripts/
├── train_fairseq2_from_batches.py  # fairseq2 training (deterministic)
├── train_hf_single_gpu.py          # HuggingFace training (deterministic)
├── extract_fairseq2_batches.py     # Extract batches from fairseq2 data
├── compare_checkpoints_simple.py   # Compare final parameters
├── compare_training_curves.py      # Compare step-by-step losses
└── run_comparison.sh               # Orchestration script

comparison/
├── extracted_batches/              # Shared batch files
├── fairseq2_checkpoints/           # fairseq2 output
└── hf_checkpoints/                 # HuggingFace output
```

## Summary

The fairseq2 HuggingFace adapter produces **identical convergence** to native HuggingFace Transformers when:

1. ✅ Flash Attention is disabled (`attn_implementation="eager"`)
2. ✅ Full PyTorch determinism is enabled
3. ✅ Dropout is disabled (`model.eval()`)
4. ✅ Identical data ordering (same extracted batches)
5. ✅ Identical training configuration (optimizer, scheduler, seed)
6. ✅ Same GPU used for both runs

This validates that the fairseq2 HG adapter correctly wraps HuggingFace models without introducing any training differences.

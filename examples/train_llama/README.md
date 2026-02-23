# Llama 3.2 3B Instruct Training Example

This example demonstrates training Llama 3.2 3B Instruct using fairseq2's trainer infrastructure.

## Structure

```
train_llama/
├── config.yaml       # Training configuration
├── config.py         # TrainingConfig dataclass
├── data.py          # Data pipeline creation
├── train_unit.py    # Training unit for forward pass
├── eval_unit.py     # Evaluation unit
├── trainer.py       # Trainer setup and initialization
├── train.py         # Main training script
└── README.md        # This file
```

## Quick Start

### Single GPU Training
```bash
# With FP16 (recommended for GPUs)
python train.py --config config.yaml --device cuda:0

# With FP32 (safer for first run to check loss scaling)
python train.py --config config.yaml --device cuda:0 --no-amp
```

### Multi-GPU Training (FSDP)
```bash
# 4 GPUs with FP16
torchrun --nproc_per_node=4 train.py --config config.yaml --device cuda

# 4 GPUs with FP32
torchrun --nproc_per_node=4 train.py --config config.yaml --device cuda --no-amp
```

### CPU Training (Testing Only)
```bash
python train.py --config config.yaml --device cpu --no-amp
```

## Configuration

The `config.yaml` file contains all training parameters:

### Model & Dataset
- **model.name**: Model asset name (default: `llama3_2_3b_instruct`)
- **dataset.name**: Dataset asset name (default: `openeft`)

### Device & Precision
- **device.type**: Device (`cpu`, `cuda`, `cuda:0`, etc.)
- **precision.amp**: Enable automatic mixed precision (FP16)
- **precision.amp_dtype**: Data type for AMP (`float16` or `bfloat16`)

### Training Hyperparameters
- **training.learning_rate**: Learning rate (default: `1e-5`)
- **training.batch_size**: Per-device batch size (default: `2`)
- **training.num_accumulate**: Gradient accumulation steps (default: `4`)
- **training.max_num_data_epochs**: Number of training epochs (default: `3`)
- **training.max_seq_len**: Maximum sequence length (default: `512`)
- **training.weight_decay**: Weight decay for AdamW (default: `0.01`)
- **training.max_grad_norm**: Gradient clipping threshold (default: `1.0`)
- **training.warmup_steps**: LR warmup steps (default: `100`)

### Data Split
- **data.eval_split_ratio**: Fraction of data for evaluation (default: `0.1` = 10%)
- **data.seed**: Random seed (default: `42`)

### FP16 Loss Scaling
- **fp16.init_scale**: Initial loss scale (default: `65536`)
- **fp16.min_scale**: Minimum loss scale (default: `0.1`)
- **fp16.scale_window**: Steps before increasing scale (default: `1000`)

### Checkpointing
- **checkpoint.output_dir**: Directory for checkpoints (default: `./checkpoints`)
- **checkpoint.checkpoint_every_n_steps**: Save frequency (default: `100`)
- **checkpoint.keep_last_n_checkpoints**: Number of checkpoints to keep (default: `3`)

### Validation & Logging
- **validation.validate_every_n_data_epochs**: Validation frequency (default: `1`)
- **logging.publish_metrics_every_n_steps**: Metric logging frequency (default: `50`)

## Command-Line Overrides

You can override configuration values from the command line:

```bash
python train.py \
    --config config.yaml \
    --output-dir ./my_checkpoints \
    --device cuda:1 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --max-num-epochs 5
```

Available overrides:
- `--output-dir`: Output directory
- `--device`: Device type
- `--no-amp`: Disable mixed precision
- `--batch-size`: Batch size
- `--num-accumulate`: Gradient accumulation
- `--learning-rate`: Learning rate
- `--max-num-epochs`: Number of epochs
- `--max-seq-len`: Maximum sequence length

## Architecture

- **Model**: Llama 3.2 3B Instruct (loaded from fairseq2 HuggingFace assets)
- **Training Task**: Causal language modeling (next-token prediction)
- **Precision**: FP16 mixed precision with dynamic loss scaling
- **Data Pipeline**: Optimized fairseq2 data pipeline with prefetching
- **Parallelism**: FSDP (Fully Sharded Data Parallel) for multi-GPU training
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Validation**: Periodic evaluation on held-out data (10% split)

## Metrics

The training loop tracks:
- **train_loss**: Training cross-entropy loss
- **train_ppl**: Training perplexity (exp(loss))
- **eval_loss**: Evaluation cross-entropy loss
- **eval_ppl**: Evaluation perplexity
- **grad_norm**: Gradient norm (for monitoring)
- **num_tokens**: Tokens processed

## Requirements

- fairseq2 with Llama 3.2 assets
- PyTorch with CUDA support (for GPU training)
- PyYAML for configuration loading
- Sufficient GPU memory (3B model requires ~12GB+ for FP16 training)

## Notes

- **Loss Scaling**: If you encounter NaN losses, try `--no-amp` first, then adjust `fp16.init_scale` in the config
- **Memory**: Reduce `batch_size` or `max_seq_len` if you run out of GPU memory
- **Checkpoints**: Checkpoints are saved to `output_dir` and include model state, optimizer state, and training progress

## Troubleshooting

### Gradient Explosion / "Loss is scaled down to minimum"

If you see errors like:
```
ERROR | fairseq2 | Overflow detected at step N, loss scale is already at minimum (0.0001)
ERROR | fairseq2 | Training failed with error: Loss is scaled down to minimum at step N
```

**Root Cause**: This means gradients are exploding (becoming inf/NaN).

**Solutions** (in order of likelihood):

1. **Verify model is loaded in FP32** (MOST COMMON)
   - Even with AMP enabled, the model should be loaded in `torch.float32`
   - The Trainer's `autocast()` handles FP16 conversion during forward pass
   - Loading in FP16 loses precision and causes gradient explosion
   - **Fixed in current version** - model always loads in FP32

2. **Lower the learning rate**
   - Try reducing by 10x: `learning_rate: 1.0e-6` (from `1.0e-5`)
   - Especially important for fine-tuning pretrained models

3. **Increase effective batch size**
   - Increase `batch_size` (if GPU memory allows)
   - Increase `num_accumulate` (gradient accumulation steps)
   - Larger batches = more stable gradients

4. **Strengthen gradient clipping**
   - Current default is `max_grad_norm: 1.0`
   - Try lower values: `0.5` or `0.1`

5. **Check your data**
   - Ensure inputs are properly normalized
   - Check for extreme values or outliers

### NaN Losses with FP16

If you see NaN values for `train_loss`, `eval_loss`, or perplexity:

1. **Try FP32 first**: Run with `--no-amp` to verify the setup works
2. **Check loss scale**: The `fp16.min_scale` in config should be at least `0.0001` (fairseq2 default)
   - Too low values (e.g., `1e-10`) cause FP16 underflow
   - The default `init_scale` of `65536.0` (2^16) is usually good
3. **Gradient explosion**: If gradients explode, reduce `fp16.init_scale` or `learning_rate`
4. **Monitor scale changes**: Check logs for "increasing/decreasing loss scale" messages

### Zero Timing Values

If `wall_time` and `total_time` show as `0.0`:
- This was a bug in the initial version - the `wall_watch` Stopwatch wasn't started
- Fixed in the current version by calling `wall_watch.start()` before passing to Trainer

### Out of Memory

If you run out of GPU memory:
1. Reduce `batch_size` (default: 2)
2. Reduce `max_seq_len` (default: 512)
3. Increase `num_accumulate` to maintain effective batch size
4. Use gradient checkpointing (requires model modification)

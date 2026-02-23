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

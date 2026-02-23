# Training Recipe: Llama 3.2 3B Instruct

This directory contains a complete training recipe for fine-tuning Llama 3.2 3B Instruct using fairseq2's trainer infrastructure.

## Features

### Architecture
- **Model**: Llama 3.2 3B Instruct (loaded from HuggingFace via fairseq2 assets)
- **Task**: Causal language modeling with next-token prediction
- **Dataset**: Generic instruction dataset (synthetic examples for demonstration)

### Training Optimizations
- **FP16 Mixed Precision**: Automatic mixed precision training with dynamic loss scaling
- **Gradient Accumulation**: Configurable accumulation steps for larger effective batch sizes
- **FSDP**: Fully Sharded Data Parallel for multi-GPU training
- **Optimized Data Pipeline**:
  - Sharded data loading across ranks
  - Prefetching for overlapping I/O and computation
  - Bucketing by sequence length for efficient batching

### Infrastructure
- **Checkpointing**: Automatic checkpoint saving with configurable frequency
- **Metrics**: TensorBoard logging for training metrics
- **Gang Abstraction**: fairseq2's gang system for distributed training
- **Multi-Device Support**: CPU, single-GPU, and multi-GPU execution

## Installation

Ensure you have fairseq2 installed with CUDA support (for GPU training):

```bash
# Install fairseq2 (if not already installed)
pip install fairseq2

# Or install from source (see CLAUDE.md for detailed instructions)
```

## Usage

### Single GPU Training (Recommended for Testing)

```bash
# FP16 training on GPU 0
python train_llama3_2_3b_instruct.py \
    --output-dir ./checkpoints \
    --device cuda:0 \
    --batch-size 2 \
    --num-accumulate 4 \
    --learning-rate 1e-5 \
    --max-num-steps 1000
```

### Multi-GPU Training with FSDP

```bash
# 4-GPU training with FSDP
torchrun --nproc_per_node=4 train_llama3_2_3b_instruct.py \
    --output-dir ./checkpoints \
    --device cuda \
    --batch-size 2 \
    --num-accumulate 4 \
    --learning-rate 1e-5 \
    --max-num-steps 1000
```

### CPU Training (For Testing Only)

```bash
# FP32 training on CPU (very slow, for debugging only)
python train_llama3_2_3b_instruct.py \
    --output-dir ./checkpoints \
    --device cpu \
    --no-amp \
    --batch-size 1 \
    --num-accumulate 2 \
    --max-num-steps 100
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | Path | `./checkpoints` | Output directory for checkpoints and logs |
| `--device` | str | `cuda` | Device (cpu, cuda, cuda:0, etc.) |
| `--no-amp` | flag | False | Disable FP16 mixed precision |
| `--batch-size` | int | 2 | Per-device batch size |
| `--num-accumulate` | int | 4 | Gradient accumulation steps |
| `--learning-rate` | float | 1e-5 | Learning rate |
| `--max-num-steps` | int | 1000 | Maximum training steps |
| `--max-seq-len` | int | 512 | Maximum sequence length |

## Effective Batch Size

The effective batch size is calculated as:

```
effective_batch_size = batch_size × num_accumulate × num_gpus
```

For example, with `--batch-size 2 --num-accumulate 4` on 4 GPUs:
- Effective batch size = 2 × 4 × 4 = 32

## Training Configuration

The recipe uses the following default hyperparameters optimized for LLM fine-tuning:

```python
learning_rate = 1e-5          # Conservative LR for fine-tuning
weight_decay = 0.01           # L2 regularization
max_grad_norm = 1.0           # Gradient clipping
warmup_steps = 100            # Linear warmup
optimizer = AdamW(β1=0.9, β2=0.95)  # Standard LLM optimizer
lr_schedule = CosineAnnealing # Cosine decay to 10% of initial LR
```

## Data Pipeline

The recipe demonstrates fairseq2's data pipeline with:

1. **Synthetic Dataset**: For demonstration, uses synthetic instruction examples
   - In production, replace with actual dataset loading from fairseq2 assets

2. **Pipeline Operations**:
   - Sharding across data parallel ranks
   - Shuffling with buffer of 1000 examples
   - Tokenization using model's tokenizer
   - Sequence length bucketing
   - Batching with configurable batch size
   - Prefetching for performance

3. **Replace with Real Data**:
   ```python
   # In create_data_pipeline(), replace sample_instructions with:
   from fairseq2.datasets import load_dataset
   dataset = load_dataset("your_dataset_name", split="train")
   pipeline = dataset.read_batch(gangs)
   ```

## Monitoring

### TensorBoard

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir ./checkpoints/tensorboard
```

Tracked metrics:
- `train_loss`: Cross-entropy loss
- `train_ppl`: Perplexity (exp of loss)
- `num_tokens`: Number of tokens processed
- `grad_norm`: Gradient norm
- `learning_rate`: Current learning rate

### Checkpoints

Checkpoints are saved every 100 steps to `./checkpoints/checkpoints/`:
- `step_100/`: Checkpoint at step 100
- `step_200/`: Checkpoint at step 200
- etc.

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training state (step number, epoch, etc.)
- Data reader state (for resuming from exact position)

## Customization

### Adjusting Hyperparameters

Edit the `TrainingConfig` dataclass in the script:

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5  # Change learning rate
    batch_size: int = 2          # Change batch size
    max_num_steps: int = 1000    # Change training duration
    # ... other parameters
```

### Using a Different Model

Change the `model_name`:

```python
model_name: str = "llama3_1_8b_instruct"  # Or any other fairseq2 model
```

### Custom Dataset

Replace the synthetic data pipeline in `create_data_pipeline()`:

```python
def create_data_pipeline(...):
    # Load your dataset
    from fairseq2.datasets import load_dataset
    dataset = load_dataset("your_dataset_name")

    # Build your custom pipeline
    pipeline = dataset.read_batch(gangs).map(...).batch(...)
    return pipeline
```

### Custom Training Logic

Modify the `CausalLMTrainUnit.process_batch()` method:

```python
def process_batch(self, batch, metric_bag):
    # Your custom forward pass
    # Your custom loss computation
    # Your custom metrics
    return loss, num_targets
```

## Performance Tips

### Memory Optimization

1. **Reduce batch size**: `--batch-size 1`
2. **Increase gradient accumulation**: `--num-accumulate 8`
3. **Reduce sequence length**: `--max-seq-len 256`
4. **Enable FSDP reshard**: Already enabled in recipe

### Speed Optimization

1. **Increase batch size** (if memory allows): `--batch-size 4`
2. **Use multiple GPUs**: `torchrun --nproc_per_node=8`
3. **Increase prefetch**: Modify `prefetch(num_batches=8)` in pipeline
4. **Mixed precision**: FP16 is enabled by default

### Convergence

1. **Adjust learning rate**: Try `--learning-rate 5e-6` or `--learning-rate 2e-5`
2. **Increase warmup**: Modify `warmup_steps` in config
3. **Adjust weight decay**: Modify `weight_decay` in config
4. **Gradient clipping**: Modify `max_grad_norm` in config

## Architecture Details

### Gang Abstraction

fairseq2 uses "gangs" instead of raw PyTorch distributed:

```python
gangs = get_default_gangs(device)  # Get gangs for device
gangs = create_fsdp_gangs(gangs)   # Create FSDP-specific gangs

# gangs.root: Root gang (all processes)
# gangs.dp: Data parallel gang
# gangs.tp: Tensor parallel gang (if enabled)
# gangs.rdp: Replica data parallel gang (for FSDP)
```

### FSDP Configuration

The model is wrapped with FSDP for multi-GPU training:

```python
model = to_data_parallel(
    model,
    gangs,
    fsdp_wrap_granularity="layer",      # Wrap per layer
    fsdp_reshard_after_forward=True,    # Reshard for memory
)
```

### FP16 Loss Scaling

Dynamic loss scaling prevents underflow in FP16:

```python
fp16_loss_scaler = StandardFloat16LossScaler(
    gangs,
    init_scale=2.0 ** 16,      # Initial scale
    scale_window=2000,         # Steps before increasing scale
)
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Increase `--num-accumulate`
- Reduce `--max-seq-len`
- Enable gradient checkpointing (modify model config)

### Slow Training

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`
- Increase `--batch-size` if memory allows
- Verify data pipeline prefetching is working

### NaN Loss

- Reduce `--learning-rate`
- Check FP16 loss scaling (increase `init_scale`)
- Verify input data doesn't have NaN values
- Enable gradient clipping (already enabled in recipe)

### Model Loading Fails

- Verify model asset exists: `python -c "from fairseq2.models import load_model; load_model('llama3_2_3b_instruct')"`
- Check internet connection (for HuggingFace downloads)
- Verify fairseq2 asset card configuration

## License

This recipe is part of the fairseq2 project and follows the same BSD-style license.

## References

- fairseq2 Documentation: https://facebookresearch.github.io/fairseq2/
- fairseq2 GitHub: https://github.com/facebookresearch/fairseq2
- Llama 3.2: https://ai.meta.com/llama/

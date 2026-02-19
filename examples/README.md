# fairseq2 Examples

This directory contains example scripts demonstrating fairseq2 usage.

## Simple Gemma SFT Training (`simple_gemma_sft.py`)

Ultra simple script that demonstrates:
- Loading a Gemma model using fairseq2's HuggingFace integration
- Creating supervised fine-tuning (SFT) training dataset with **20 examples**
- Training loop with **batching** (batch size 4, 5 batches per epoch)
- **Checkpoint management** with automatic resume using fairseq2's `StandardCheckpointManager`
- Multi-GPU support with FSDP

### Prerequisites

Make sure fairseq2 is installed and the virtual environment is activated:

```bash
source .venv/bin/activate
```

### Usage

**CPU mode** (slow, for testing only):
```bash
python examples/simple_gemma_sft.py --device cpu
```

**GPU mode** (recommended):
```bash
python examples/simple_gemma_sft.py --device cuda
```

**With custom output directory and batch size**:
```bash
python examples/simple_gemma_sft.py --device cuda --output-dir my_checkpoint --batch-size 2
```

**Multi-GPU mode with FSDP** (requires 2+ GPUs):
```bash
torchrun --nproc_per_node=2 examples/simple_gemma_sft.py --device cuda
```

**Multi-GPU with 4 GPUs**:
```bash
torchrun --nproc_per_node=4 examples/simple_gemma_sft.py --device cuda
```

**Custom training parameters**:
```bash
python examples/simple_gemma_sft.py --device cuda --num-epochs 2 --learning-rate 1e-5 --batch-size 8
```

### What It Does

1. Loads the `google/gemma-3-1b-it` model using `load_causal_lm()` from fairseq2's HG integration
2. **Sets up CheckpointManager** and checks for existing checkpoints
3. **Automatically resumes from checkpoint** if one exists in the output directory
4. **Automatically detects multi-GPU setup** and initializes FSDP (Fully Sharded Data Parallel) if using `torchrun`
5. Creates 20 simple question-answer training pairs
6. **Creates DataLoader** with configurable batch size (default: 4, resulting in 5 batches per epoch)
7. Trains the model for 1 epoch (default) with very conservative settings for stability
8. **Saves checkpoint** at the end using fairseq2's `StandardCheckpointManager`
9. Tests the model by generating a response (single-GPU mode only)

### Checkpointing Features

The script uses fairseq2's `StandardCheckpointManager` for robust checkpointing:
- **Automatic resume**: If a checkpoint exists in `--output-dir`, training automatically resumes from the last step
- **Saves model and optimizer state** for proper resumption
- **Distributed-safe**: Works correctly in multi-GPU setups
- **Checkpoints saved to**: `{output_dir}/checkpoints/step_{N}/`

**Example - Resume Training:**
```bash
# First run (trains and saves checkpoint)
python examples/simple_gemma_sft.py --output-dir my_run --num-epochs 1

# Resume from checkpoint (automatically detects and loads)
python examples/simple_gemma_sft.py --output-dir my_run --num-epochs 2
```

### Multi-GPU Support

The script automatically enables **FSDP** when running with `torchrun` and multiple GPUs:
- Uses fairseq2's gang abstraction for distributed training
- Applies FSDP using `to_fsdp2()` and `apply_fsdp_to_hg_transformer_lm()`
- Shards model parameters across GPUs to save memory (layer-wise granularity)
- Only rank 0 prints output to avoid clutter
- Automatically cleans up distributed resources on exit

**Note on device display:** When using FSDP, each rank sees its local device (e.g., cuda:0 from rank 0's perspective, cuda:1 from rank 1's perspective). The model is actually sharded across all GPUs - this is expected FSDP behavior.

**Generation with FSDP:** The script skips the generation test when FSDP is enabled because FSDP-wrapped models require all ranks to participate in forward passes. For inference with FSDP models, you would need a dedicated inference script where all ranks participate.

To disable FSDP in multi-GPU mode:
```bash
torchrun --nproc_per_node=2 examples/simple_gemma_sft.py --device cuda --no-fsdp
```

### Training Stability

The script uses conservative settings for stable training:
- **fp32 precision** instead of fp16 (more stable without loss scaling)
- **Very low learning rate** (5e-6) to prevent instability
- **Gradient clipping** (max norm 1.0) to prevent exploding gradients
- **NaN detection** with early stopping if training becomes unstable

You can experiment with more aggressive settings:
```bash
python examples/simple_gemma_sft.py --device cuda --num-epochs 5 --learning-rate 1e-5
```

### Expected Output

You should see:
- Model loading progress
- Checkpoint detection (if resuming)
- 20 training examples, 5 batches per epoch (with batch_size=4)
- Per-step loss output (e.g., "Step 1 (batch 1/5) | Loss: 3.5477")
- Checkpoint saving confirmation
- A generated response at the end (single-GPU only)

Note: This is an educational example showing the basics. For production training, use fairseq2's full `Trainer` class with proper data pipelines, validation, early stopping, etc.

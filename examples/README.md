# fairseq2 Examples

This directory contains example scripts demonstrating how to use the fairseq2 library for various tasks.

## LLaMA Inference (`llama_inference.py`)

A simple inference script demonstrating text generation using fairseq2's LLaMA 3.2 1B model loaded from the asset system.

### Features

- Loads LLaMA 3.2 models from fairseq2's asset system
- Automatic checkpoint downloading and caching
- Multi-device support: CPU, single-GPU, and multi-GPU via gang abstraction
- Sampling-based text generation with Top-P (nucleus) sampling
- High-level `TextCompleter` API for easy text generation

### Quick Start

```bash
# CPU mode
python examples/llama_inference.py --device cpu

# Single GPU mode
python examples/llama_inference.py --device cuda:0

# Multi-GPU mode (2 GPUs using torchrun)
torchrun --nproc_per_node=2 examples/llama_inference.py --device cuda
```

### Usage

```bash
python examples/llama_inference.py [OPTIONS]
```

**Options:**

- `--model MODEL`: Model name from fairseq2 assets (default: `llama3_2_1b_instruct`)
- `--device DEVICE`: Device specification - `cpu`, `cuda`, or `cuda:N` (default: `cpu`)
- `--max-gen-len N`: Maximum generation length in tokens (default: `128`)
- `--temperature T`: Sampling temperature, higher = more random (default: `0.6`)
- `--top-p P`: Top-p (nucleus) sampling parameter (default: `0.9`)

**Examples:**

```bash
# Use the base model instead of instruct
python examples/llama_inference.py --model llama3_2_1b --device cuda:0

# Generate longer sequences with higher temperature
python examples/llama_inference.py \
    --device cuda:0 \
    --max-gen-len 256 \
    --temperature 0.8 \
    --top-p 0.95

# Multi-GPU inference with 4 GPUs
torchrun --nproc_per_node=4 examples/llama_inference.py \
    --device cuda \
    --max-gen-len 200
```

### Available LLaMA 3.2 Models

The following LLaMA 3.2 models are available in fairseq2 assets:

| Model Name | Description | Size |
|------------|-------------|------|
| `llama3_2_1b` | Base model | 1B parameters |
| `llama3_2_1b_instruct` | Instruction-tuned model | 1B parameters |
| `llama3_2_3b` | Larger base model | 3B parameters |
| `llama3_2_3b_instruct` | Larger instruction-tuned model | 3B parameters |

Other LLaMA models (3.1, 3.0, 2.0) are also available. Use `fairseq2 assets list` to see all available models.

### How It Works

The script demonstrates key fairseq2 patterns:

1. **Model Loading**: Uses `load_model()` to load models by name from the asset system
2. **Tokenizer Loading**: Uses `load_tokenizer()` to load the corresponding tokenizer
3. **Gang Abstraction**: Uses `setup_default_gang()` for distributed execution
4. **Generation**: Uses `SamplingSequenceGenerator` with `TopPSampler` for text generation
5. **Text API**: Uses `TextCompleter` for high-level text completion

### Requirements

- fairseq2 (installed)
- PyTorch
- (Optional) CUDA for GPU execution
- (Optional) Multiple GPUs and `torchrun` for multi-GPU execution

### Datasets

The fairseq2 asset system includes the `openeft` dataset (dataset_family: `generic_instruction`) which can be used for instruction-following tasks. Dataset integration is not included in this example but can be added using `fairseq2.datasets.DatasetHub`.

### Multi-GPU Execution

For multi-GPU execution, use PyTorch's `torchrun` utility:

```bash
# 2 GPUs
torchrun --nproc_per_node=2 examples/llama_inference.py --device cuda

# 4 GPUs with custom port
torchrun --nproc_per_node=4 --master_port=29500 examples/llama_inference.py --device cuda
```

The script uses fairseq2's gang abstraction, which handles distributed setup automatically when launched with `torchrun`.

## Additional Examples

More examples will be added to this directory demonstrating:
- Fine-tuning with fairseq2's Trainer
- Custom data pipeline usage
- Model evaluation and benchmarking
- Integration with vLLM for faster inference

---

For more information, see the [fairseq2 documentation](https://github.com/facebookresearch/fairseq2).

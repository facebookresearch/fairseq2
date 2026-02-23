# Qwen Inference Example

This example demonstrates text generation using fairseq2's Qwen 2.5 7B Instruct model.

## Structure

```
qwen_inference/
├── config.yaml        # Configuration file
├── model.py          # Model and tokenizer loading
├── generator.py      # Text generator setup
├── inference.py      # Main inference script
└── README.md         # This file
```

## Quick Start

### CPU Mode
```bash
python inference.py --config config.yaml
```

### Single GPU
```bash
python inference.py --config config.yaml --device cuda:0
```

### Multi-GPU (using torchrun)
```bash
torchrun --nproc_per_node=2 inference.py --config config.yaml --device cuda
```

## Configuration

Edit `config.yaml` to customize:
- **model.name**: Model asset name (default: `qwen25_7b_instruct`)
- **device.type**: Device type (`cpu`, `cuda`, `cuda:0`, etc.)
- **generation.max_gen_len**: Maximum tokens to generate
- **generation.temperature**: Sampling temperature (0.0-1.0)
- **generation.top_p**: Nucleus sampling parameter (0.0-1.0)
- **prompts**: List of prompts for inference

## Requirements

- fairseq2 installed with model assets
- PyTorch with appropriate CUDA support (for GPU inference)
- PyYAML for configuration loading

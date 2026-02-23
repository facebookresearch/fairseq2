# fairseq2 Examples

This directory contains modularized example scripts demonstrating fairseq2's capabilities for model inference and training.

## Available Examples

### 1. Qwen Inference (`qwen_inference/`)

Text generation using Qwen 2.5 7B Instruct model.

**Quick start:**
```bash
cd qwen_inference
python inference.py --config config.yaml --device cuda:0
```

**Features:**
- Model loading from fairseq2 assets
- Text completion with sampling
- CPU, single-GPU, and multi-GPU support

[See detailed documentation](qwen_inference/README.md)

---

### 2. Llama Training (`train_llama/`)

Full training pipeline for Llama 3.2 3B Instruct with causal language modeling.

**Quick start:**
```bash
cd train_llama
python train.py --config config.yaml --device cuda:0
```

**Features:**
- Complete training setup with FSDP
- FP16 mixed precision training
- Data pipeline with train/eval splits
- Checkpoint management
- Metric logging

[See detailed documentation](train_llama/README.md)

---

### 3. Test Training Setup (`test_train_setup/`)

Verification script to test that training components initialize correctly.

**Quick start:**
```bash
cd test_train_setup
python test_setup.py --config config.yaml
```

**Features:**
- Validates all training components
- No actual training (fast check)
- Useful for debugging setup issues

[See detailed documentation](test_train_setup/README.md)

---

## Structure

Each example is organized as a Python package with:
- **`config.yaml`**: YAML configuration file
- **Module files**: Separated concerns (model, data, training, etc.)
- **Main script**: Entry point for running the example
- **`README.md`**: Detailed documentation

## General Usage Pattern

All examples follow a consistent pattern:

1. **Edit configuration**: Modify `config.yaml` for your use case
2. **Run the script**: Execute the main Python script with `--config` flag
3. **Override settings**: Use command-line arguments to override config values

Example:
```bash
python script.py --config config.yaml --device cuda:0 --batch-size 4
```

## Requirements

- fairseq2 installed with model assets
- PyTorch with appropriate CUDA support
- PyYAML (`pip install pyyaml`)

## Legacy Scripts

The following legacy scripts are still available in the parent directory but are **deprecated** in favor of the modularized versions:

- ❌ `qwen_inference.py` → Use `qwen_inference/` instead
- ❌ `train_llama3_2_3b_instruct.py` → Use `train_llama/` instead
- ❌ `test_train_setup.py` → Use `test_train_setup/` instead

## Tips

### GPU Memory Issues
If you encounter out-of-memory errors:
- Reduce `batch_size` in config.yaml
- Reduce `max_seq_len` in config.yaml
- Enable gradient accumulation (`num_accumulate > 1`)

### Device Selection
- **CPU**: `--device cpu` (slow, for testing only)
- **Single GPU**: `--device cuda:0` (or `cuda:1`, etc.)
- **Multi-GPU**: `--device cuda` with `torchrun --nproc_per_node=N`

### Multi-GPU Training
Use `torchrun` for distributed training:
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

The gang abstraction handles process group initialization automatically.

## Contributing

When adding new examples:
1. Create a new subdirectory
2. Follow the modular structure (config.yaml, separate modules, main script)
3. Include a comprehensive README.md
4. Add an entry to this main README

## Support

For issues or questions:
- Check individual example READMEs
- See fairseq2 documentation: https://facebookresearch.github.io/fairseq2/
- Report issues: https://github.com/facebookresearch/fairseq2/issues

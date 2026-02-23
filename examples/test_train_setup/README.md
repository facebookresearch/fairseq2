# Test Training Setup

This script verifies that all training components can be created successfully without running actual training.

## Structure

```
test_train_setup/
├── config.yaml      # Test configuration
├── test_setup.py    # Test script
└── README.md        # This file
```

## Quick Start

```bash
python test_setup.py --config config.yaml
```

## What It Tests

This script verifies:
- Model loading from fairseq2 assets
- Tokenizer loading
- Data pipeline creation (train and eval splits)
- Training unit initialization
- Evaluation unit initialization
- Validator creation
- Trainer initialization

All components are created but no actual training is performed. This is useful for:
- Verifying installation
- Testing configuration changes
- Debugging setup issues

## Configuration

Edit `config.yaml` to customize test parameters:
- **training.output_dir**: Directory for temporary checkpoints
- **training.device**: Device type (`cpu` recommended for testing)
- **training.amp**: Whether to use mixed precision (disable for CPU)
- **training.batch_size**: Per-device batch size
- **training.num_accumulate**: Gradient accumulation steps

## Note

This test depends on the `train_llama` example being present in the parent directory, as it imports the training configuration and setup functions from there.

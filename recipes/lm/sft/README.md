# Language Model Supervised Fine-Tuning (SFT)

This directory contains the implementation for supervised fine-tuning of language models using fairseq2. The SFT recipe allows you to fine-tune pre-trained language models on instruction-following datasets.

## Overview

The SFT recipe is designed to finetune language models to follow instructions by learning from prompt-response pairs.

This recipe supports both Llama and Qwen model families.

## Key Features

- **Flexible Data Loading**: Support for both local files and Hugging Face Hub datasets
- **Chat Mode**: Built-in support for chat templates (compatible with Hugging Face chat templates)
- **Dynamic Batching**: Static batching or length-based batching for efficient training
- **Distributed Training**: Full support for multi-GPU and multi-node training

## Quick Start

### Basic Usage

```bash
# Run with a pre-configured Llama setup
python -m recipes.lm.sft --config-name recipes/lm/sft/configs/llama3_2_1b_gsm8k.yaml

# Run with a pre-configured Qwen setup
python -m recipes.lm.sft --config-name recipes/lm/sft/configs/qwen3_4b_gsm8k.yaml

# Run with custom config file")
python -m recipes.lm.sft --config-file path/to/your/config.yaml
```

### Custom Configuration

Create a YAML configuration file with your settings (note that we have examples in `sft/configs/`):

```yaml
dataset:
  path: "path/to/your/dataset"  # Local path or HuggingFace Hub URI (hg://username/dataset)
  train_split: "train"
  valid_split: "validation"
  batch_size: 16
  max_seq_len: 2048
  chat_mode: false

model:
  family: "llama"
  name: "llama3_2_1b"

tokenizer:
  family: "llama" 
  name: "llama3_2_1b"

regime:
  num_steps: 10000
  checkpoint_every_n_steps: 1000
  validate_every_n_steps: 1000
```

#### Examples

##### LLaMA 3.2 1B on GSM8K
```bash
python -m fairseq2.recipes.lm.sft --config-file sft/configs/llama3_2_1b_gsm8k.yaml
```

##### LLaMA 3.2 1B Instruct on GSM8K  
```bash
python -m fairseq2.recipes.lm.sft --config-file sft/configs/llama3_2_1b_instruct_gsm8k.yaml
```

## Data Format

Your JSONL files should contain entries with `src` (source/prompt) and `tgt` (target/response) fields:

```json
{"src": "What is the capital of France?", "tgt": "The capital of France is Paris."}
{"src": "Explain photosynthesis in simple terms.", "tgt": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."}
```

To use huggingface chat wrappers, set `chat_mode=True` in your config.

## Common Configs

### Dataset

- `path`: Path to dataset (local directory/file or HuggingFace Hub URI with `hg://` prefix)
- `train_split`/`valid_split`: Names of the train/validation splits
- `batch_size`: Fixed batch size (alternative to dynamic batching)
- `max_num_tokens`: Maximum tokens per batch for dynamic batching
- `max_seq_len`: Maximum sequence length (longer sequences are dropped)
- `min_seq_len`: Minimum sequence length
- `chat_mode`: Enable chat template processing
- `source_encode_mode`/`target_encode_mode`: Tokenizer encoding modes

### Training

- `num_steps`: Total number of training steps
- `checkpoint_every_n_steps`: Checkpoint saving frequency
- `validate_every_n_steps`: Validation frequency
- `keep_last_n_checkpoints`: Number of recent checkpoints to retain

## File Structure

- `__main__.py`: Entry point for the SFT recipe
- `recipe.py`: Core recipe implementation with training logic
- `config.py`: Configuration classes and default settings
- `dataset.py`: Dataset loading and processing logic
- `utils.py`: Utility functions for data handling
- `configs/`: Pre-configured YAML files for common setups



## Advanced Usage

### Using HuggingFace Hub Datasets

```yaml
dataset:
  path: "hg://username/dataset-name"
  # The dataset will be automatically downloaded and cached
```

### Custom Batching Strategies

```yaml
dataset:
  # Fixed batch size
  batch_size: 32
  
  # OR dynamic batching by token count
  max_num_tokens: 8192
  max_seq_len: 2048
```


## Troubleshooting

1. **Out of Memory**: Reduce `batch_size` or `max_num_tokens`
2. **Data Format Issues**: Ensure JSONL files have correct `src`/`tgt` or `chat` fields
3. **Path Issues**: Use absolute paths or ensure relative paths are correct from working directory
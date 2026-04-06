# Multimodal Generate Recipe

Multimodal inference recipe for vision-language models. Supports image and video
inputs via HuggingFace model integration with fairseq2's recipe framework.

## Overview

This recipe loads a HuggingFace multimodal model (e.g., Gemma 3) with its
processor, reads a JSONL dataset of chat-style messages containing text, image,
and video content, and runs `.generate()` to produce responses. Video inputs are
automatically expanded into evenly sampled frames.

## Quick Start

```bash
# Single GPU
python -m recipes.multimodal.generate \
  --config-file recipes/multimodal/generate/configs/gemma3_4b_it.yaml ./output_dir

# Multi-GPU
torchrun --nproc_per_node=8 -m recipes.multimodal.generate \
  --config-file recipes/multimodal/generate/configs/gemma3_4b_it.yaml ./output_dir
```

## Dataset Format

The dataset is a JSONL file where each line contains a chat-style message list.
Content blocks support `text`, `image`, and `video` types:

```jsonl
{"id": "1", "messages": [{"role": "user", "content": [{"type": "video", "url": "path/to/video.mp4"}, {"type": "text", "text": "Describe this video."}]}]}
{"id": "2", "messages": [{"role": "user", "content": [{"type": "image", "url": "path/to/image.jpg"}, {"type": "text", "text": "What is in this image?"}]}]}
{"id": "3", "messages": [{"role": "user", "content": "What is the capital of France?"}]}
```

- **`type: video`** — Frames are extracted with `decord` and expanded into
  individual image blocks for the model's chat template.
- **`type: image`** — Loaded directly as a PIL image.
- **`type: text`** — Passed through as-is.
- Plain string content is also supported for text-only queries.

## Configuration

```yaml
model:
  hf_name: "google/gemma-3-4b-it"   # HuggingFace model identifier
  dtype: bfloat16                     # Model dtype
  trust_remote_code: true             # Trust remote code for custom architectures

dataset:
  family: "multimodal_generate"
  batch_size: 1
  config_overrides:
    paths:
      - "path/to/dataset.jsonl"

generation:
  max_new_tokens: 512                 # Maximum tokens to generate per example
  do_sample: false                    # Greedy decoding by default
  temperature: 1.0                    # Sampling temperature (when do_sample=true)
  top_p: 1.0                         # Nucleus sampling threshold

video:
  num_frames: 8                       # Number of frames to sample per video
```

## Output

Results are written to `<output_dir>/output/` with per-rank files:

- `rank_0.txt` — Human-readable output with prompts and responses.
- `rank_0.jsonl` — Machine-readable JSONL with `id`, `prompt`, and `response` fields.

## Dependencies

Requires `decord` for video frame extraction:

```bash
pip install decord
```

## Architecture

The recipe registers multimodal models (e.g., Gemma 3) as special HuggingFace
model classes via `register_hg_model_class()`, following the same pattern used
for Qwen 2.5 Omni in `fairseq2.models.hg.factory`. This routes model loading
through `_load_special_model()`, which loads the correct conditional generation
class (e.g., `Gemma3ForConditionalGeneration`) with its processor attached.

```
recipes/multimodal/generate/
├── __main__.py    # Entry point
├── config.py      # Config dataclasses
├── dataset.py     # JSONL dataset reader
├── video.py       # Video frame extraction and message rewriting
├── recipe.py      # Recipe and Task implementation
└── configs/
    └── gemma3_4b_it.yaml
```

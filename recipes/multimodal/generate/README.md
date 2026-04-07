# Multimodal Generate Recipe

Multimodal inference recipe for vision-language models. Supports image, video,
and audio inputs via HuggingFace model integration with fairseq2's recipe
framework. Model-family-specific logic is handled by pluggable handlers.

## Supported Models

| Model Family | Handler | Notes |
|---|---|---|
| Gemma 3 | `gemma3` | Images + video (frame extraction via decord) |
| Qwen 2.5 Omni | `qwen_omni` | Images + video + audio (native processor support) |

## Quick Start

```bash
# Gemma 3 — single GPU
python -m recipes.multimodal.generate \
  --config-file recipes/multimodal/generate/configs/gemma3_4b_it.yaml ./output_dir

# Qwen 2.5 Omni — single GPU
python -m recipes.multimodal.generate \
  --config-file recipes/multimodal/generate/configs/qwen2_5_omni_7b.yaml ./output_dir

# Multi-GPU
torchrun --nproc_per_node=8 -m recipes.multimodal.generate \
  --config-file recipes/multimodal/generate/configs/gemma3_4b_it.yaml ./output_dir
```

## Dataset Format

The dataset is a JSONL file where each line contains a chat-style message list.
Content blocks support `text`, `image`, `video`, and `audio` types:

```jsonl
{"id": "1", "messages": [{"role": "user", "content": [{"type": "video", "url": "path/to/video.mp4"}, {"type": "text", "text": "Describe this video."}]}]}
{"id": "2", "messages": [{"role": "user", "content": [{"type": "image", "url": "path/to/image.jpg"}, {"type": "text", "text": "What is in this image?"}]}]}
{"id": "3", "messages": [{"role": "user", "content": "What is the capital of France?"}]}
```

- **`type: video`** — Gemma 3: frames extracted with `decord` and expanded into
  image blocks. Qwen 2.5 Omni: passed directly to the processor.
- **`type: image`** — Loaded as a PIL image.
- **`type: audio`** — Supported by Qwen 2.5 Omni; ignored by Gemma 3.
- **`type: text`** — Passed through as-is.
- Plain string content is also supported for text-only queries.

## Configuration

```yaml
model:
  hf_name: "google/gemma-3-4b-it"   # HuggingFace model identifier
  dtype: bfloat16                     # Model dtype
  trust_remote_code: true             # Trust remote code for custom architectures
  handler: auto                       # "auto", "gemma3", or "qwen_omni"

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

The `handler` field controls model-family-specific logic. Set to `"auto"`
(default) to detect from `hf_name`, or specify explicitly.

## Output

Results are written to `<output_dir>/output/` with per-rank files:

- `rank_0.txt` — Human-readable output with prompts and responses.
- `rank_0.jsonl` — Machine-readable JSONL with `id`, `prompt`, and `response` fields.

## Dependencies

Requires `decord` for video frame extraction (Gemma 3):

```bash
pip install decord
```

## Architecture

```
recipes/multimodal/generate/
├── __main__.py    # Entry point
├── config.py      # Config dataclasses
├── dataset.py     # JSONL dataset reader
├── video.py       # Video frame extraction (used by Gemma 3 handler)
├── recipe.py      # Recipe and Task — delegates to handler
├── handlers/
│   ├── __init__.py    # ModelHandler protocol + get_handler() factory
│   ├── gemma3.py      # Gemma 3 handler
│   └── qwen_omni.py   # Qwen 2.5 Omni handler
├── configs/
│   ├── gemma3_4b_it.yaml
│   └── qwen2_5_omni_7b.yaml
└── data/
    └── dummy_video_dataset.jsonl
```

Model-family-specific logic (input preparation, generation kwargs, decoding) is
encapsulated in handler classes implementing the `ModelHandler` protocol. The
recipe dispatches to the appropriate handler based on the `handler` config field.

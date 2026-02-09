#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Inspecting LAuReL Parameter Shapes ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path
from huggingface_hub import snapshot_download

hf_cache = Path.home() / ".cache/huggingface/hub"

# Find model directory
model_dirs = list(hf_cache.glob("models--google--gemma-3n-E2B-it/snapshots/*"))
if not model_dirs:
    print("Downloading model...")
    model_path = snapshot_download(
        "google/gemma-3n-E2B-it",
        allow_patterns=["*.safetensors"],
        cache_dir=hf_cache.parent,
    )
    model_dir = Path(model_path)
else:
    model_dir = model_dirs[0]

# Load shard 3 which has LAuReL parameters
shard_file = model_dir / "model-00003-of-00003.safetensors"
if not shard_file.exists():
    raise FileNotFoundError(f"Shard not found: {shard_file}")

with safe_open(shard_file, framework="pt", device="cpu") as f:
    # Inspect layer 0 LAuReL parameters
    linear_left = f.get_tensor("model.language_model.layers.0.laurel.linear_left.weight")
    linear_right = f.get_tensor("model.language_model.layers.0.laurel.linear_right.weight")
    post_norm = f.get_tensor("model.language_model.layers.0.laurel.post_laurel_norm.weight")

    print(f"Layer 0 LAuReL shapes:")
    print(f"  linear_left:     {linear_left.shape}")
    print(f"  linear_right:    {linear_right.shape}")
    print(f"  post_laurel_norm: {post_norm.shape}")

    # Check a few more layers to confirm consistency
    print(f"\nLayer 5 LAuReL shapes:")
    linear_left_5 = f.get_tensor("model.language_model.layers.5.laurel.linear_left.weight")
    linear_right_5 = f.get_tensor("model.language_model.layers.5.laurel.linear_right.weight")
    print(f"  linear_left:     {linear_left_5.shape}")
    print(f"  linear_right:    {linear_right_5.shape}")

    # Get model_dim from input_layernorm
    input_norm = f.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    print(f"\nModel dimension (from input_layernorm): {input_norm.shape[0]}")

    # Infer LAuReL rank
    rank = linear_left.shape[0]
    model_dim = input_norm.shape[0]
    print(f"\nLAuReL Configuration:")
    print(f"  rank: {rank}")
    print(f"  model_dim: {model_dim}")
    print(f"  Architecture: linear_left: ({rank}, {model_dim}), linear_right: ({model_dim}, {rank})")
    print(f"  → Low-rank factorization of ({model_dim}, {model_dim}) residual")

PYEOF

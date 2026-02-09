#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Discovering Which Layers Have LAuReL ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path
from huggingface_hub import snapshot_download
import re

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

# Scan all shards for LAuReL layers
laurel_layers = set()
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            match = re.search(r"layers\.(\d+)\.laurel", key)
            if match:
                layer_idx = int(match.group(1))
                laurel_layers.add(layer_idx)

print(f"Found LAuReL in {len(laurel_layers)} layers:")
print(f"  Layers: {sorted(laurel_layers)}")

# Analyze pattern (local vs global)
total_layers = 30  # E2B has 30 layers
non_laurel = set(range(total_layers)) - laurel_layers
print(f"\nLayers WITHOUT LAuReL ({len(non_laurel)}):")
print(f"  Layers: {sorted(non_laurel)}")

# Check if pattern is 4:1 local:global
if len(non_laurel) == 6 and non_laurel == {4, 9, 14, 19, 24, 29}:
    print("\n✓ Pattern: Every 5th layer (0-indexed: 4, 9, 14, 19, 24, 29) is global (no LAuReL)")
    print("  → 24 local layers with LAuReL, 6 global layers without")
elif len(laurel_layers) > 0:
    # Inspect one LAuReL layer to get shapes
    first_laurel_layer = min(laurel_layers)
    print(f"\n=== LAuReL Structure (Layer {first_laurel_layer}) ===")

    for shard_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            try:
                linear_left = f.get_tensor(f"model.language_model.layers.{first_laurel_layer}.laurel.linear_left.weight")
                linear_right = f.get_tensor(f"model.language_model.layers.{first_laurel_layer}.laurel.linear_right.weight")
                post_norm = f.get_tensor(f"model.language_model.layers.{first_laurel_layer}.laurel.post_laurel_norm.weight")

                print(f"  linear_left:      {linear_left.shape}")
                print(f"  linear_right:     {linear_right.shape}")
                print(f"  post_laurel_norm: {post_norm.shape}")

                rank = linear_left.shape[0]
                model_dim = linear_right.shape[0]
                print(f"\n  Inferred config:")
                print(f"    rank: {rank}")
                print(f"    model_dim: {model_dim}")
                break
            except:
                continue
PYEOF

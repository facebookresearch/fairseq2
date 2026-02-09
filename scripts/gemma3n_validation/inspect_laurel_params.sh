#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Inspecting LAuReL Parameters in HF Checkpoint ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path
import re
import sys

hf_cache = Path.home() / ".cache/huggingface/hub"

# Find model directory
model_dirs = list(hf_cache.glob("models--google--gemma-3n-E2B-it/snapshots/*"))
if not model_dirs:
    print("✗ Model not found in cache. Downloading...")
    # Download model first
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(
        "google/gemma-3n-E2B-it",
        allow_patterns=["*.safetensors"],
        cache_dir=hf_cache.parent,
    )
    model_dir = Path(model_path)
else:
    model_dir = model_dirs[0]

print(f"Model directory: {model_dir}\n")

# Load all shards and find LAuReL-related keys
laurel_keys = []
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "laurel" in key.lower():
                laurel_keys.append((key, shard_file.name))

if laurel_keys:
    print(f"✓ Found {len(laurel_keys)} LAuReL-related keys:\n")
    for key, shard in laurel_keys:
        print(f"  {shard:30s}  {key}")
else:
    print("✗ No LAuReL-related keys found")

# Also check for PLE and other advanced features
print("\n=== Checking for PLE (Per-Layer Embeddings) ===")
ple_keys = []
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "per_layer" in key or "embed_tokens_per_layer" in key:
                ple_keys.append((key, shard_file.name))

if ple_keys:
    print(f"✓ Found {len(ple_keys)} PLE-related keys:\n")
    for key, shard in ple_keys[:10]:  # Show first 10
        print(f"  {shard:30s}  {key}")
    if len(ple_keys) > 10:
        print(f"  ... and {len(ple_keys) - 10} more")
else:
    print("✗ No PLE-related keys found")

# Check for additional normalizations
print("\n=== Checking for Additional Normalizations ===")
norm_keys = []
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if re.search(r"(pre_feedforward|post_feedforward|post_per_layer)", key):
                norm_keys.append((key, shard_file.name))

if norm_keys:
    print(f"✓ Found {len(norm_keys)} additional norm keys:\n")
    for key, shard in norm_keys[:10]:
        print(f"  {shard:30s}  {key}")
    if len(norm_keys) > 10:
        print(f"  ... and {len(norm_keys) - 10} more")
else:
    print("✗ No additional norm keys found")

# Sample one layer's full structure
print("\n=== Layer 0 Complete Structure ===")
layer0_keys = []
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".layers.0." in key:
                layer0_keys.append(key)

for key in sorted(layer0_keys):
    print(f"  {key}")
PYEOF

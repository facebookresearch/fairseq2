#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Inspecting QK Normalization Parameters ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path

hf_cache = Path.home() / ".cache/huggingface/hub"
model_dirs = list(hf_cache.glob("models--google--gemma-3n-E2B-it/snapshots/*"))
model_dir = model_dirs[0] if model_dirs else None

if not model_dir:
    print("Model not cached")
    exit(1)

# Search for q_norm and k_norm parameters
qk_norm_keys = []
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "q_norm" in key or "k_norm" in key:
                tensor = f.get_tensor(key)
                qk_norm_keys.append((key, tensor.shape, shard_file.name))

print(f"Found {len(qk_norm_keys)} QK normalization parameters:\n")

# Show first few layers
for key, shape, shard in qk_norm_keys[:10]:
    print(f"  {shard:30s}  {key:80s}  {shape}")

if len(qk_norm_keys) > 10:
    print(f"  ... and {len(qk_norm_keys) - 10} more")

# Check layer 0 specifically
print("\n=== Layer 0 QK Norm Details ===")
layer0_keys = [k for k, s, sh in qk_norm_keys if "layers.0." in k]
for key, shape, shard in [(k, s, sh) for k, s, sh in qk_norm_keys if "layers.0." in k]:
    print(f"  {key:80s}  {shape}")

# Get the actual shape
if layer0_keys:
    for shard_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            if layer0_keys[0] in f.keys():
                q_norm = f.get_tensor(layer0_keys[0])
                print(f"\nQK Norm configuration:")
                print(f"  Dimension: {q_norm.shape[0]}")
                print(f"  Type: RMSNorm (inferred from Gemma3n architecture)")
                break
PYEOF

#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Inspecting PLE Parameter Shapes ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path

hf_cache = Path.home() / ".cache/huggingface/hub"
model_dirs = list(hf_cache.glob("models--google--gemma-3n-E2B-it/snapshots/*"))

if not model_dirs:
    print("Model not cached on login node")
    exit(0)

model_dir = model_dirs[0]

# Find PLE parameters
ple_params = {}
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "embed_tokens_per_layer" in key or "per_layer_input" in key or "per_layer_projection" in key:
                if "layers.0." in key or "embed_tokens_per_layer" in key:
                    ple_params[key] = f.get_tensor(key).shape

print("PLE Parameters:\n")

# Shared embedding
if "model.language_model.embed_tokens_per_layer.weight" in ple_params:
    shape = ple_params["model.language_model.embed_tokens_per_layer.weight"]
    print(f"Shared across all layers:")
    print(f"  embed_tokens_per_layer: {shape}")
    vocab_size, hidden_size = shape
    print(f"  → vocab_size: {vocab_size}, hidden_size: {hidden_size}")

# Layer 0 specific
print(f"\nLayer 0 PLE components:")
for key, shape in sorted(ple_params.items()):
    if "layers.0." in key:
        param_name = key.split(".")[-2]
        print(f"  {param_name:30s}: {shape}")

print("\nPLE Architecture inference:")
print("  1. Shared embed_tokens_per_layer lookup table")
print("  2. Per-layer gating mechanism (per_layer_input_gate)")
print("  3. Per-layer projection (per_layer_projection)")
print("  4. Post-PLE normalization (post_per_layer_input_norm)")
PYEOF

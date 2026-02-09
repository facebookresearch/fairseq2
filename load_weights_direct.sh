#!/bin/bash
# Load Gemma3n weights directly without instantiating full model
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Installing timm ==="
$VENV/bin/pip install -q timm

echo ""
echo "=== Loading HuggingFace Weights ==="
$PYTHON << 'PYEOF'
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os

model_id = "google/gemma-3n-E2B-it"

print(f"Downloading {model_id} files...")
model_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])

print(f"✓ Downloaded to: {model_path}")

# Find safetensors files
safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
print(f"✓ Found {len(safetensor_files)} safetensor files")

# Load first file to inspect keys
first_file = os.path.join(model_path, safetensor_files[0])
print(f"\nInspecting {safetensor_files[0]}...")

with safe_open(first_file, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"  Keys in this shard: {len(keys)}")

    # Show first 20 keys
    print("\nFirst 20 keys:")
    for key in keys[:20]:
        tensor = f.get_tensor(key)
        print(f"  {key}: {tensor.shape}")

    # Check for text model patterns
    text_keys = [k for k in keys if 'text_model' in k or 'language_model' in k or 'model.layers' in k]
    if text_keys:
        print(f"\n✓ Found {len(text_keys)} text model keys")
    else:
        # Show all top-level prefixes
        prefixes = set(k.split('.')[0] for k in keys)
        print(f"\nTop-level prefixes: {sorted(prefixes)}")

print("\n✓ Weight inspection complete")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Weights loaded and inspected"

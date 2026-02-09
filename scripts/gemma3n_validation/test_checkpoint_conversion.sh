#!/bin/bash
# Test checkpoint conversion from HuggingFace to fairseq2
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Loading and Converting HF Checkpoint ==="
$PYTHON << 'PYEOF'
import sys
sys.path.insert(0, '/home/aerben/repos/fairseq2/src')

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os

from fairseq2.models.gemma3n import convert_gemma3n_state_dict, get_gemma3n_e2b_config, create_gemma3n_model

model_id = "google/gemma-3n-E2B-it"
config = get_gemma3n_e2b_config()

print(f"Downloading {model_id}...")
model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])

# Load all safetensors files
safetensor_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
print(f"✓ Found {len(safetensor_files)} shard(s)")

# Merge all shards into one state dict
hf_state_dict = {}
for shard_file in safetensor_files:
    shard_path = os.path.join(model_path, shard_file)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            hf_state_dict[key] = f.get_tensor(key)

print(f"✓ Loaded {len(hf_state_dict)} parameters")

# Show sample keys before conversion
print("\nSample HF keys:")
for key in sorted(hf_state_dict.keys())[:5]:
    print(f"  {key}: {hf_state_dict[key].shape}")

# Convert to fairseq2 format
print("\nConverting to fairseq2 format...")
try:
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    print(f"✓ Converted {len(fs2_state_dict)} parameters")

    # Show sample converted keys
    print("\nSample fairseq2 keys:")
    for key in sorted(fs2_state_dict.keys())[:5]:
        print(f"  {key}: {fs2_state_dict[key].shape}")

    # Create model and try loading
    print("\nCreating fairseq2 model...")
    model = create_gemma3n_model(config, device=torch.device("cpu"), dtype=torch.float32)

    print("Loading converted weights...")
    missing, unexpected = model.load_state_dict(fs2_state_dict, strict=False)

    print(f"✓ Weights loaded")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    if missing:
        print("\nFirst 10 missing keys:")
        for key in sorted(missing)[:10]:
            print(f"  - {key}")

    if unexpected:
        print("\nFirst 10 unexpected keys:")
        for key in sorted(unexpected)[:10]:
            print(f"  - {key}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Checkpoint conversion tested"

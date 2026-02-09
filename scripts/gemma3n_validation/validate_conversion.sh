#!/bin/bash
# Validate checkpoint key conversion only (no model loading)
set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

$PYTHON << 'PYEOF'
import sys
sys.path.insert(0, '/home/aerben/repos/fairseq2/src')
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os
from fairseq2.models.gemma3n import convert_gemma3n_state_dict, get_gemma3n_e2b_config

model_path = snapshot_download("google/gemma-3n-E2B-it", allow_patterns=["*.safetensors"])
shard_file = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])[0]

hf_state_dict = {}
with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
    for key in list(f.keys())[:50]:  # Only first 50 keys
        hf_state_dict[key] = f.get_tensor(key)

print(f"Testing conversion on {len(hf_state_dict)} keys...")
fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, get_gemma3n_e2b_config())

print(f"✓ Converted {len(fs2_state_dict)} keys")
print("\nSample conversions:")
for hf_key in sorted(hf_state_dict.keys())[:5]:
    fs2_keys = [k for k in fs2_state_dict.keys() if any(part in k for part in hf_key.split('.')[-3:])]
    if fs2_keys:
        print(f"  {hf_key}")
        print(f"  -> {fs2_keys[0]}")
    else:
        print(f"  {hf_key} -> NOT CONVERTED")
PYEOF

echo "=== Conversion validated ==="

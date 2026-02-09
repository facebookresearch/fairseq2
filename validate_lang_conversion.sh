#!/bin/bash
# Validate language_model key conversion
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

# Load from shard 3 which has language_model.layers
shard_file = "model-00003-of-00003.safetensors"
hf_state_dict = {}

with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
    # Get first 30 language_model keys
    lang_keys = [k for k in f.keys() if 'language_model.layers' in k][:30]
    for key in lang_keys:
        hf_state_dict[key] = f.get_tensor(key)

print(f"Testing conversion on {len(hf_state_dict)} language_model keys...\n")
fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, get_gemma3n_e2b_config())

print(f"✓ Converted {len(fs2_state_dict)} keys\n")
print("Sample conversions:")
for hf_key in sorted(hf_state_dict.keys())[:10]:
    # Find corresponding fs2 key
    if hf_key in fs2_state_dict:
        fs2_key = hf_key  # Passthrough
        status = "(passthrough)"
    else:
        # Try to find converted key
        layer_num = hf_key.split('.')[3] if 'layers.' in hf_key else None
        param_type = hf_key.split('.')[-2:] if '.' in hf_key else None
        fs2_candidates = [k for k in fs2_state_dict.keys()
                          if layer_num and layer_num in k and any(p in k for p in param_type)]
        fs2_key = fs2_candidates[0] if fs2_candidates else "NOT FOUND"
        status = "✓" if fs2_candidates else "✗"

    print(f"  {status} {hf_key}")
    if fs2_key != "NOT FOUND" and fs2_key != hf_key:
        print(f"     -> {fs2_key}")

print("\n=== Conversion validated ===")
PYEOF

#!/bin/bash
# Simple key conversion demo
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
shard_file = "model-00003-of-00003.safetensors"

hf_state_dict = {}
with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
    # Get representative keys
    for key in f.keys():
        if 'layers.1.mlp' in key or 'layers.1.input_layernorm' in key or 'layers.1.post_attention' in key:
            hf_state_dict[key] = f.get_tensor(key)
            if len(hf_state_dict) >= 10:
                break

print("Converting keys...\n")
fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, get_gemma3n_e2b_config())

print("HF -> fairseq2 conversions:\n")
for hf_key in sorted(hf_state_dict.keys()):
    # Find matching fs2 key
    matching = [k for k in fs2_state_dict.keys() if 'layers.1' in k and hf_key.split('.')[-1] in k]
    if matching:
        print(f"✓ {hf_key}")
        print(f"  -> {matching[0]}\n")
    else:
        print(f"✗ {hf_key} (not converted)\n")

print(f"Total: {len(hf_state_dict)} HF keys -> {len(fs2_state_dict)} fs2 keys")
PYEOF

#!/bin/bash
# Test HuggingFace checkpoint loading and key conversion
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Loading HuggingFace Gemma3n-E2B Model ==="
$PYTHON << 'PYEOF'
import sys
sys.path.insert(0, '/home/aerben/repos/fairseq2/src')

import torch
from transformers import AutoModelForCausalLM

model_id = "google/gemma-3n-E2B-it"

print(f"Loading {model_id} (downloading ~4GB, will be cached)...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"✓ Model loaded")
print(f"  Total params: {sum(p.numel() for p in hf_model.parameters()) / 1e9:.2f}B")

# Get state dict
state_dict = hf_model.state_dict()
print(f"  State dict keys: {len(state_dict)}")

# Show first 20 keys
print("\nFirst 20 keys:")
for i, key in enumerate(sorted(state_dict.keys())[:20]):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"  {key}: {shape}")

# Check for text model structure
text_model_keys = [k for k in state_dict.keys() if 'text_model' in k or 'language_model' in k or 'model.layers' in k]
if text_model_keys:
    print(f"\n✓ Found {len(text_model_keys)} text model keys")
    print(f"  Sample: {text_model_keys[0]}")
else:
    print("\n✗ No text model keys found - showing all key patterns:")
    prefixes = set(k.split('.')[0] for k in state_dict.keys())
    print(f"  Top-level prefixes: {prefixes}")

PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "HF model loaded and inspected"

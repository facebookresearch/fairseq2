#!/bin/bash
# Just show what keys exist in the checkpoint
set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

$PYTHON << 'PYEOF'
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os

model_path = snapshot_download("google/gemma-3n-E2B-it", allow_patterns=["*.safetensors"])
shard_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])

print(f"Found {len(shard_files)} shards\n")

for shard_file in shard_files:
    with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"{shard_file}: {len(keys)} keys")

        # Show first 10 keys
        for key in keys[:10]:
            print(f"  {key}")

        # Check for embed_tokens
        embed_keys = [k for k in keys if 'embed_tokens' in k]
        if embed_keys:
            print(f"  ✓ Found embed_tokens: {embed_keys}")
        print()
PYEOF

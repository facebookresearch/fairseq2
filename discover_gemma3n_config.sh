#!/bin/bash
# Inspect HuggingFace Gemma3n config attributes
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Discovering Gemma3n Config Attributes ==="
$PYTHON << 'PYEOF'
import torch
from transformers import AutoConfig

model_id = "google/gemma-3n-E2B-it"

print(f"Loading config from {model_id}...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"\nConfig class: {type(config).__name__}")
print(f"Model type: {config.model_type}")

print("\nAll config attributes:")
for attr in sorted(dir(config)):
    if not attr.startswith('_') and not callable(getattr(config, attr)):
        value = getattr(config, attr)
        if not isinstance(value, (dict, list)) or len(str(value)) < 100:
            print(f"  {attr}: {value}")

print("\n✓ Config inspection complete")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Config attributes discovered"

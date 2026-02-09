#!/bin/bash
# Download and test HuggingFace Gemma3n model
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Checking available Gemma models ==="
$PYTHON << 'PYEOF'
from huggingface_hub import list_models

# Search for Gemma3n models
models = list_models(search="gemma-3", limit=20)
gemma3_models = [m.id for m in models if 'gemma-3' in m.id.lower()]

print("Available Gemma-3 models:")
for model in gemma3_models[:10]:
    print(f"  - {model}")
PYEOF

echo ""
echo "=== Testing if google/gemma-2-2b works as proxy ==="
$PYTHON << 'PYEOF'
from transformers import AutoConfig
import sys

try:
    # Try gemma-2-2b as a starting point
    config = AutoConfig.from_pretrained("google/gemma-2-2b", trust_remote_code=True)
    print(f"✓ Config loaded: {config.model_type}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Model search completed"

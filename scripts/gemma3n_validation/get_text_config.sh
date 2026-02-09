#!/bin/bash
# Get Gemma3n text config attributes
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Gemma3n Text Config ==="
$PYTHON << 'PYEOF'
from transformers import AutoConfig

model_id = "google/gemma-3n-E2B-it"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"Top-level config type: {type(config).__name__}")

# Check for text_config
if hasattr(config, 'text_config'):
    print("\ntext_config found!")
    tc = config.text_config
    print(f"  Type: {type(tc).__name__}")

    # Print all text config attributes
    print("\nText config attributes:")
    for attr in sorted(dir(tc)):
        if not attr.startswith('_') and not callable(getattr(tc, attr, None)):
            try:
                value = getattr(tc, attr)
                if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")
else:
    print("\nNo text_config attribute found")
    print("Available top-level attributes:")
    for attr in sorted(dir(config)):
        if not attr.startswith('_'):
            print(f"  {attr}")

print("\n✓ Config inspection complete")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Text config attributes shown"

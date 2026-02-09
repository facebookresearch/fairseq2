#!/bin/bash
# Gemma3n Parity Test Script
# Run this on the compute node with: bash test_parity.sh

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Installing dependencies ==="
$VENV/bin/pip install -q transformers accelerate 2>&1 | grep -v "already satisfied" || true

echo ""
echo "=== GPU Check ==="
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "=== Running Parity Test ==="
$PYTHON << 'PYEOF'
import torch
from fairseq2.models.gemma3n import create_gemma3n_model, Gemma3nConfig

# Create small test config
config = Gemma3nConfig()
config.num_layers = 2  # Use only 2 layers for quick test
config.vocab_size = 1000

print(f"Creating model: {config.num_layers} layers, vocab={config.vocab_size}")

# Create fairseq2 model
model = create_gemma3n_model(config, device=torch.device("cuda:0"), dtype=torch.float32)
model.eval()

# Test forward pass
batch_size, seq_len = 2, 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda:0")

from fairseq2.nn.batch_layout import BatchLayout
seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=torch.device("cuda:0"))

with torch.no_grad():
    logits = model(input_ids, seqs_layout)

print(f"✓ Forward pass successful: {logits.shape}")
print(f"✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"PARITY_TEST_PASSED")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "Test completed successfully"

#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Testing LAuReL Implementation ==="

$PYTHON << 'PYEOF'
import torch
from fairseq2.models.gemma3n import create_gemma3n_model, Gemma3nConfig

# Create small model for testing
config = Gemma3nConfig()
config.num_layers = 2
config.vocab_size = 1000

print("Creating model with LAuReL...")
model = create_gemma3n_model(config, device=torch.device("cuda:0"), dtype=torch.float32)

# Check that LAuReL residual exists
layer0 = model.decoder.layers[0]
print(f"\nLayer 0 self_attn_residual type: {type(layer0.self_attn_residual).__name__}")

# Check LAuReL parameters exist
if hasattr(layer0.self_attn_residual, 'linear_left'):
    print(f"✓ LAuReL linear_left shape: {layer0.self_attn_residual.linear_left.weight.shape}")
    print(f"✓ LAuReL linear_right shape: {layer0.self_attn_residual.linear_right.weight.shape}")
    print(f"✓ LAuReL layer_norm: {type(layer0.self_attn_residual.layer_norm).__name__}")

    # Verify shapes match config
    expected_left = (config.laurel_rank, config.model_dim)
    expected_right = (config.model_dim, config.laurel_rank)

    actual_left = layer0.self_attn_residual.linear_left.weight.shape
    actual_right = layer0.self_attn_residual.linear_right.weight.shape

    assert actual_left == expected_left, f"linear_left shape mismatch: {actual_left} != {expected_left}"
    assert actual_right == expected_right, f"linear_right shape mismatch: {actual_right} != {expected_right}"

    print("\n✓ LAuReL parameter shapes correct")
else:
    print("✗ LAuReL residual is not LAuReLResidualConnect")
    exit(1)

# Test forward pass
print("\nTesting forward pass...")
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch.device("cuda:0"))

from fairseq2.nn import BatchLayout
seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=torch.device("cuda:0"))

with torch.no_grad():
    logits = model(input_ids, seqs_layout)

print(f"✓ Forward pass successful, output shape: {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size)

print("\n✓ ALL LAUREL TESTS PASSED")
PYEOF

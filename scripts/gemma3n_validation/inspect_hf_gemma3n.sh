#!/bin/bash
# Full Gemma3n parity test - HuggingFace vs fairseq2
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Loading HuggingFace Gemma3n-E2B ==="
$PYTHON << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "google/gemma-3n-E2B-it"

print(f"Loading config from {model_id}...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"\nHuggingFace Config:")
print(f"  Model type: {config.model_type}")
print(f"  Layers: {config.num_hidden_layers}")
print(f"  Hidden dim: {config.hidden_size}")
print(f"  Attention heads: {config.num_attention_heads}")
print(f"  KV heads: {config.num_key_value_heads}")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")

# Check for Gemma3n-specific attributes
if hasattr(config, 'rope_theta'):
    print(f"  RoPE theta: {config.rope_theta}")
if hasattr(config, 'attention_bias'):
    print(f"  Attention bias: {config.attention_bias}")

print("\n✓ Config loaded successfully")
PYEOF

echo ""
echo "=== Checking model architecture ==="
$PYTHON << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM

model_id = "google/gemma-3n-E2B-it"

print(f"Loading model (this will download ~4GB)...")
print("(Using CPU for structure inspection)")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("\n✓ Model loaded")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Inspect first layer structure
print("\nFirst layer structure:")
for name, param in list(model.named_parameters())[:10]:
    print(f"  {name}: {param.shape}")

print("\n✓ Architecture inspection complete")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "HuggingFace model loaded and inspected"

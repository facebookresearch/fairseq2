#!/bin/bash
# Gemma3n-E2B Architecture Parity Test
# Run this on the compute node

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python
REPO=/home/aerben/repos/fairseq2

echo "=== Testing E2B Config Match ==="
$PYTHON << 'PYEOF'
import sys
sys.path.insert(0, '/home/aerben/repos/fairseq2/src')

from transformers import AutoConfig
from fairseq2.models.gemma3n import get_gemma3n_e2b_config

# Load HF config
hf_config = AutoConfig.from_pretrained("google/gemma-3n-E2B-it", trust_remote_code=True)
hf_tc = hf_config.text_config

# Load fairseq2 config
fs2_config = get_gemma3n_e2b_config()

print("Config Comparison:")
print(f"  Layers:        HF={hf_tc.num_hidden_layers:2d}  FS2={fs2_config.num_layers:2d}  {'✓' if hf_tc.num_hidden_layers == fs2_config.num_layers else '✗'}")
print(f"  Hidden dim:    HF={hf_tc.hidden_size:4d}  FS2={fs2_config.model_dim:4d}  {'✓' if hf_tc.hidden_size == fs2_config.model_dim else '✗'}")
print(f"  Attn heads:    HF={hf_tc.num_attention_heads:2d}  FS2={fs2_config.num_attn_heads:2d}  {'✓' if hf_tc.num_attention_heads == fs2_config.num_attn_heads else '✗'}")
print(f"  KV heads:      HF={hf_tc.num_key_value_heads:2d}  FS2={fs2_config.num_key_value_heads:2d}  {'✓' if hf_tc.num_key_value_heads == fs2_config.num_key_value_heads else '✗'}")
print(f"  Vocab size:    HF={hf_tc.vocab_size:6d}  FS2={fs2_config.vocab_size:6d}  {'✓' if hf_tc.vocab_size == fs2_config.vocab_size else '✗'}")
print(f"  Max seq len:   HF={hf_tc.max_position_embeddings:5d}  FS2={fs2_config.max_seq_len:5d}  {'✓' if hf_tc.max_position_embeddings == fs2_config.max_seq_len else '✗'}")
print(f"  Sliding win:   HF={hf_tc.sliding_window:3d}  FS2={fs2_config.sliding_window:3d}  {'✓' if hf_tc.sliding_window == fs2_config.sliding_window else '✗'}")
print(f"  RMS eps:       HF={hf_tc.rms_norm_eps}  FS2={fs2_config.rms_norm_eps}  {'✓' if hf_tc.rms_norm_eps == fs2_config.rms_norm_eps else '✗'}")

# Check all match
all_match = (
    hf_tc.num_hidden_layers == fs2_config.num_layers and
    hf_tc.hidden_size == fs2_config.model_dim and
    hf_tc.num_attention_heads == fs2_config.num_attn_heads and
    hf_tc.num_key_value_heads == fs2_config.num_key_value_heads and
    hf_tc.vocab_size == fs2_config.vocab_size and
    hf_tc.max_position_embeddings == fs2_config.max_seq_len and
    hf_tc.sliding_window == fs2_config.sliding_window and
    hf_tc.rms_norm_eps == fs2_config.rms_norm_eps
)

if all_match:
    print("\n✓ ALL CONFIG VALUES MATCH")
else:
    print("\n✗ CONFIG MISMATCH DETECTED")
    sys.exit(1)
PYEOF

echo ""
echo "=== Creating fairseq2 E2B Model ==="
$PYTHON << 'PYEOF'
import sys
sys.path.insert(0, '/home/aerben/repos/fairseq2/src')

import torch
from fairseq2.models.gemma3n import create_gemma3n_model, get_gemma3n_e2b_config

config = get_gemma3n_e2b_config()
print(f"Creating {config.num_layers}-layer model...")

model = create_gemma3n_model(config, device=torch.device("cuda:0"), dtype=torch.float32)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {total_params / 1e9:.2f}B parameters")

# Test forward pass
batch_size, seq_len = 2, 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda:0")

from fairseq2.nn.batch_layout import BatchLayout
seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=torch.device("cuda:0"))

with torch.no_grad():
    logits = model(input_ids, seqs_layout)

print(f"✓ Forward pass: {logits.shape}")
print(f"✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
PYEOF

echo ""
echo "=== COPY THIS RESULT ==="
echo "✓ E2B config matches HuggingFace"
echo "✓ fairseq2 model creates and runs successfully"

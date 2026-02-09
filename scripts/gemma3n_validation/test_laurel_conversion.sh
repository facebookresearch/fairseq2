#!/bin/bash
set -e
VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Testing LAuReL Checkpoint Conversion ==="

$PYTHON << 'PYEOF'
from safetensors import safe_open
from pathlib import Path
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.models.gemma3n import Gemma3nConfig

hf_cache = Path.home() / ".cache/huggingface/hub"

# Find model directory
model_dirs = list(hf_cache.glob("models--google--gemma-3n-E2B-it/snapshots/*"))
if not model_dirs:
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(
        "google/gemma-3n-E2B-it",
        allow_patterns=["*.safetensors"],
        cache_dir=hf_cache.parent,
    )
    model_dir = Path(model_path)
else:
    model_dir = model_dirs[0]

# Load LAuReL parameters from layer 0
hf_state = {}

# Search all shards for layer 0 LAuReL parameters
for shard_file in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        layer0_laurel_keys = [k for k in keys if "layers.0.laurel" in k]

        if layer0_laurel_keys:
            print(f"Found layer 0 LAuReL parameters in {shard_file.name}")
            for key in layer0_laurel_keys:
                hf_state[key] = f.get_tensor(key)
            break

if not hf_state:
    print("ERROR: Could not find layer 0 LAuReL parameters in any shard")
    exit(1)

print("HuggingFace LAuReL keys (layer 0):")
for key in sorted(hf_state.keys()):
    print(f"  {key:70s} {hf_state[key].shape}")

# Convert to fairseq2 format
config = Gemma3nConfig(num_layers=30)  # E2B config
fs2_state = convert_gemma3n_state_dict(hf_state, config)

print("\nfairseq2 LAuReL keys (layer 0):")
for key in sorted(fs2_state.keys()):
    print(f"  {key:70s} {fs2_state[key].shape}")

# Verify conversion
expected_mappings = {
    "model.language_model.layers.0.laurel.linear_left.weight":
        "decoder.layers.0.self_attn_residual.linear_left.weight",
    "model.language_model.layers.0.laurel.linear_right.weight":
        "decoder.layers.0.self_attn_residual.linear_right.weight",
    "model.language_model.layers.0.laurel.post_laurel_norm.weight":
        "decoder.layers.0.self_attn_residual.layer_norm.weight",
}

print("\nVerifying conversions:")
for hf_key, fs2_key in expected_mappings.items():
    if fs2_key in fs2_state:
        print(f"  ✓ {hf_key}")
        print(f"    → {fs2_key}")

        # Verify shapes match
        assert hf_state[hf_key].shape == fs2_state[fs2_key].shape, \
            f"Shape mismatch: {hf_state[hf_key].shape} != {fs2_state[fs2_key].shape}"

        # Verify values match (conversion should preserve values)
        import torch
        assert torch.allclose(hf_state[hf_key], fs2_state[fs2_key]), \
            f"Values don't match for {hf_key}"
    else:
        print(f"  ✗ Missing: {fs2_key}")
        exit(1)

print("\n✓ ALL LAUREL CHECKPOINT CONVERSIONS VERIFIED")
PYEOF

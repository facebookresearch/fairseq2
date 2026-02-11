"""Test 8: Verify activation_sparsity is now set correctly after fix."""

import torch
from transformers import AutoModelForCausalLM
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

print("=" * 80)
print("TEST 8: Verify Activation Sparsity Configuration")
print("=" * 80)

# Check sparsity values for all layers
print("\nActivation Sparsity by Layer:")
print("-" * 80)
print(f"{'Layer':<8} {'HF Sparsity':<15} {'FS2 Sparsity':<15} {'Match':<10}")
print("-" * 80)

all_match = True
for layer_idx in range(config.num_layers):
    hf_layer = hf_model.model.language_model.layers[layer_idx]
    fs2_layer = fs2_model.decoder.layers[layer_idx]

    hf_sparsity = hf_layer.mlp.activation_sparsity
    fs2_sparsity = fs2_layer.ffn.activation_sparsity

    match = "✅" if abs(hf_sparsity - fs2_sparsity) < 1e-6 else "❌"
    if abs(hf_sparsity - fs2_sparsity) >= 1e-6:
        all_match = False

    # Print every 5th layer + first 10 + last 5
    if layer_idx < 10 or layer_idx >= config.num_layers - 5 or layer_idx % 5 == 0:
        print(f"{layer_idx:<8} {hf_sparsity:<15.2f} {fs2_sparsity:<15.2f} {match:<10}")

print("-" * 80)
if all_match:
    print("✅ ALL layers have matching activation_sparsity")
else:
    print("❌ Some layers have mismatched activation_sparsity")

print("\nExpected pattern:")
print("  Layers 0-9: 0.95 sparsity (95% of neurons zeroed)")
print(f"  Layers 10-{config.num_layers-1}: 0.00 sparsity (no sparsity)")

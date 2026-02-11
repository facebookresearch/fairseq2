"""Test 13: Compare global layer FFN (GLU) step-by-step."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

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
print("TEST 13: Global Layer FFN (GLU) Step-by-Step Comparison")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randn(1, 4, 2048, device=device, dtype=dtype)
print(f"Test input shape: {test_input.shape}\n")

hf_mlp = hf_model.model.language_model.layers[4].mlp
fs2_ffn = fs2_model.decoder.layers[4].ffn

print(f"Layer 4 (Global Layer) FFN:")
print(f"  HF type: {type(hf_mlp).__name__}")
print(f"  FS2 type: {type(fs2_ffn).__name__}")
print()

print(f"HF config:")
print(f"  hidden_size: {hf_mlp.hidden_size}")
print(f"  intermediate_size: {hf_mlp.intermediate_size}")
print(f"  activation_sparsity: {hf_mlp.activation_sparsity}")
print(f"  act_fn: {hf_mlp.act_fn}")
print()

print(f"FS2 config:")
print(f"  gate_activation: {fs2_ffn.gate_activation}")
print()

with torch.no_grad():
    print("=" * 80)
    print("Step-by-Step Comparison")
    print("=" * 80)

    print("\n1. Gate projection")
    print("-" * 80)
    hf_gate = hf_mlp.gate_proj(test_input)
    fs2_gate = fs2_ffn.gate_proj(test_input)

    diff = (hf_gate - fs2_gate).abs()
    print(f"  Shape: HF={hf_gate.shape}, FS2={fs2_gate.shape}")
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_gate.min():.4f}, {hf_gate.max():.4f}]")
    print(f"  FS2 range: [{fs2_gate.min():.4f}, {fs2_gate.max():.4f}]")

    print("\n2. Gate activation (GELU with tanh)")
    print("-" * 80)

    # HF: applies sparsity BEFORE activation
    if hf_mlp.activation_sparsity > 0:
        hf_gate_sparse = hf_mlp._gaussian_topk(hf_gate)
        print(f"  HF applies sparsity: {hf_mlp.activation_sparsity}")
        print(f"  HF sparse range: [{hf_gate_sparse.min():.4f}, {hf_gate_sparse.max():.4f}]")
    else:
        hf_gate_sparse = hf_gate
        print(f"  HF no sparsity")

    hf_gate_act = hf_mlp.act_fn(hf_gate_sparse)

    # FS2: applies activation directly (no sparsity for global layers)
    fs2_gate_act = fs2_ffn.gate_activation(fs2_gate)

    diff = (hf_gate_act - fs2_gate_act).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_gate_act.min():.4f}, {hf_gate_act.max():.4f}]")
    print(f"  FS2 range: [{fs2_gate_act.min():.4f}, {fs2_gate_act.max():.4f}]")

    print("\n3. Up projection (inner_proj)")
    print("-" * 80)
    hf_up = hf_mlp.up_proj(test_input)
    fs2_up = fs2_ffn.inner_proj(test_input)

    diff = (hf_up - fs2_up).abs()
    print(f"  Shape: HF={hf_up.shape}, FS2={fs2_up.shape}")
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_up.min():.4f}, {hf_up.max():.4f}]")
    print(f"  FS2 range: [{fs2_up.min():.4f}, {fs2_up.max():.4f}]")

    print("\n4. Gating (activation * up)")
    print("-" * 80)
    hf_gated = hf_gate_act * hf_up
    fs2_gated = fs2_up * fs2_gate_act

    diff = (hf_gated - fs2_gated).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_gated.min():.4f}, {hf_gated.max():.4f}]")
    print(f"  FS2 range: [{fs2_gated.min():.4f}, {fs2_gated.max():.4f}]")

    print("\n5. Down projection (output_proj)")
    print("-" * 80)
    hf_down = hf_mlp.down_proj(hf_gated)
    fs2_down = fs2_ffn.output_proj(fs2_gated)

    diff = (hf_down - fs2_down).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_down.min():.4f}, {hf_down.max():.4f}]")
    print(f"  FS2 range: [{fs2_down.min():.4f}, {fs2_down.max():.4f}]")

    print("\n" + "=" * 80)
    print("FULL FORWARD COMPARISON")
    print("=" * 80)

    hf_output = hf_mlp(test_input)
    fs2_output = fs2_ffn(test_input)

    diff = (hf_output - fs2_output).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")

    if diff.max() < 1e-4:
        print("  ✅ FFN outputs match")
    else:
        print(f"  ❌ FFN outputs diverge")

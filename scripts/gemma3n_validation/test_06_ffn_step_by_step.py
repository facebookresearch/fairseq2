"""Test 6: Compare HF and FS2 FFN with same weights."""

import torch
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
print("TEST 6: Compare FFN Implementations with Same Weights")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randn(1, 4, 2048, device=device, dtype=dtype)
print(f"Test input shape: {test_input.shape}\n")

hf_mlp = hf_model.model.language_model.layers[0].mlp
fs2_ffn = fs2_model.decoder.layers[0].ffn

print(f"HF MLP config:")
print(f"  hidden_size: {hf_mlp.hidden_size}")
print(f"  intermediate_size: {hf_mlp.intermediate_size}")
print(f"  activation_sparsity: {hf_mlp.activation_sparsity}")
print()

print(f"FS2 FFN config:")
print(f"  activation_sparsity: {fs2_ffn.activation_sparsity}")
print()

with torch.no_grad():
    # ========================================================================
    # Manual step-by-step comparison
    # ========================================================================

    print("Step 1: Gate projection")
    print("-" * 80)
    hf_gate = hf_mlp.gate_proj(test_input)
    fs2_gate = fs2_ffn.gate_proj(test_input)

    diff = (hf_gate - fs2_gate).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_gate.min():.4f}, {hf_gate.max():.4f}]")
    print(f"  FS2 range: [{fs2_gate.min():.4f}, {fs2_gate.max():.4f}]")
    print()

    print("Step 2: Gaussian top-k (if sparsity > 0)")
    print("-" * 80)

    if hf_mlp.activation_sparsity > 0:
        print(f"  Applying sparsity: {hf_mlp.activation_sparsity}")

        # HF Gaussian top-k
        hf_gate_sparse = hf_mlp._gaussian_topk(hf_gate)
        print(f"  HF sparse range: [{hf_gate_sparse.min():.4f}, {hf_gate_sparse.max():.4f}]")
        print(f"  HF non-zero fraction: {(hf_gate_sparse != 0).float().mean():.4f}")

        # FS2 Gaussian top-k
        fs2_gate_sparse = fs2_ffn._gaussian_topk(fs2_gate)
        print(f"  FS2 sparse range: [{fs2_gate_sparse.min():.4f}, {fs2_gate_sparse.max():.4f}]")
        print(f"  FS2 non-zero fraction: {(fs2_gate_sparse != 0).float().mean():.4f}")

        diff = (hf_gate_sparse - fs2_gate_sparse).abs()
        print(f"  Max diff after sparsity: {diff.max().item():.6e}")

        hf_gate = hf_gate_sparse
        fs2_gate = fs2_gate_sparse
    else:
        print(f"  No sparsity applied")
    print()

    print("Step 3: GELU activation")
    print("-" * 80)
    hf_act = hf_mlp.act_fn(hf_gate)
    import torch.nn.functional as F
    fs2_act = F.gelu(fs2_gate, approximate="tanh")

    diff = (hf_act - fs2_act).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_act.min():.4f}, {hf_act.max():.4f}]")
    print(f"  FS2 range: [{fs2_act.min():.4f}, {fs2_act.max():.4f}]")
    print()

    print("Step 4: Up projection")
    print("-" * 80)
    hf_up = hf_mlp.up_proj(test_input)
    fs2_up = fs2_ffn.inner_proj(test_input)

    diff = (hf_up - fs2_up).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_up.min():.4f}, {hf_up.max():.4f}]")
    print(f"  FS2 range: [{fs2_up.min():.4f}, {fs2_up.max():.4f}]")
    print()

    print("Step 5: Gating (act * up)")
    print("-" * 80)
    hf_gated = hf_act * hf_up
    fs2_gated = fs2_up * fs2_act

    diff = (hf_gated - fs2_gated).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_gated.min():.4f}, {hf_gated.max():.4f}]")
    print(f"  FS2 range: [{fs2_gated.min():.4f}, {fs2_gated.max():.4f}]")
    print()

    print("Step 6: Down projection")
    print("-" * 80)
    hf_down = hf_mlp.down_proj(hf_gated)
    fs2_down = fs2_ffn.output_proj(fs2_gated)

    diff = (hf_down - fs2_down).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  HF range: [{hf_down.min():.4f}, {hf_down.max():.4f}]")
    print(f"  FS2 range: [{fs2_down.min():.4f}, {fs2_down.max():.4f}]")
    print()

    # ========================================================================
    # Full forward comparison
    # ========================================================================
    print("=" * 80)
    print("FULL FORWARD PASS COMPARISON")
    print("=" * 80)

    hf_output = hf_mlp(test_input)
    fs2_output = fs2_ffn(test_input)

    diff = (hf_output - fs2_output).abs()
    print(f"Max diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")
    print()

    if diff.max() < 1e-5:
        print("✅ FFN implementations match")
    else:
        print(f"❌ FFN implementations diverge")
        print(f"\nHF output sample [0, 0, :5]: {hf_output[0, 0, :5]}")
        print(f"FS2 output sample [0, 0, :5]: {fs2_output[0, 0, :5]}")
        print(f"Diff [0, 0, :5]: {diff[0, 0, :5]}")

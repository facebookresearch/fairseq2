"""Test 7: Debug FS2 Gaussian top-k producing inf."""

import torch
import torch.nn.functional as F
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
print("TEST 7: Debug FS2 Gaussian Top-K Producing Inf")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randn(1, 4, 2048, device=device, dtype=dtype)

fs2_ffn = fs2_model.decoder.layers[0].ffn
hf_mlp = hf_model.model.language_model.layers[0].mlp

print(f"FS2 activation_sparsity: {fs2_ffn.activation_sparsity}")
print(f"HF activation_sparsity: {hf_mlp.activation_sparsity}")
print()

with torch.no_grad():
    # Get gate projection outputs
    fs2_gate = fs2_ffn.gate_proj(test_input)
    hf_gate = hf_mlp.gate_proj(test_input)

    print(f"Gate projection outputs:")
    print(f"  FS2 range: [{fs2_gate.min():.4f}, {fs2_gate.max():.4f}]")
    print(f"  HF range: [{hf_gate.min():.4f}, {hf_gate.max():.4f}]")
    print(f"  Diff: {(fs2_gate - hf_gate).abs().max():.6e}")
    print()

    # ========================================================================
    # Debug FS2 _gaussian_topk step by step
    # ========================================================================
    print("=" * 80)
    print("FS2 _gaussian_topk Step-by-Step Debug")
    print("=" * 80)

    inputs = fs2_gate
    activation_sparsity = fs2_ffn.activation_sparsity

    print(f"Step 1: Create target_sparsity tensor")
    target_sparsity = torch.tensor(
        activation_sparsity,
        dtype=torch.float32,
        device=inputs.device
    )
    print(f"  target_sparsity: {target_sparsity.item()}")
    print(f"  dtype: {target_sparsity.dtype}")
    print()

    print(f"Step 2: Compute std_multiplier via icdf")
    normal_dist = torch.distributions.normal.Normal(0, 1)
    std_multiplier_float = normal_dist.icdf(target_sparsity)
    print(f"  icdf(0.95) [float32]: {std_multiplier_float.item()}")
    print(f"  Is finite: {torch.isfinite(std_multiplier_float).item()}")

    std_multiplier = std_multiplier_float.to(inputs.dtype)
    print(f"  After .to({inputs.dtype}): {std_multiplier.item()}")
    print(f"  Is finite: {torch.isfinite(std_multiplier).item()}")
    print()

    print(f"Step 3: Compute statistics")
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    print(f"  inputs_mean: {inputs_mean[0, 0, 0].item():.6f}")
    print(f"  inputs_std: {inputs_std[0, 0, 0].item():.6f}")
    print(f"  Is mean finite: {torch.isfinite(inputs_mean).all().item()}")
    print(f"  Is std finite: {torch.isfinite(inputs_std).all().item()}")
    print()

    print(f"Step 4: Compute cutoff")
    cutoff = inputs_mean + inputs_std * std_multiplier
    print(f"  cutoff: {cutoff[0, 0, 0].item():.6f}")
    print(f"  Is finite: {torch.isfinite(cutoff).all().item()}")
    print()

    print(f"Step 5: Apply ReLU")
    diff = inputs - cutoff
    print(f"  inputs - cutoff range: [{diff.min():.6f}, {diff.max():.6f}]")
    print(f"  Is finite: {torch.isfinite(diff).all().item()}")

    result = F.relu(diff)
    print(f"  ReLU result range: [{result.min():.6f}, {result.max():.6f}]")
    print(f"  Is finite: {torch.isfinite(result).all().item()}")
    print(f"  Non-zero fraction: {(result != 0).float().mean():.4f}")
    print()

    # ========================================================================
    # Compare with HF
    # ========================================================================
    print("=" * 80)
    print("HF _gaussian_topk")
    print("=" * 80)

    hf_result = hf_mlp._gaussian_topk(hf_gate)
    print(f"  HF result range: [{hf_result.min():.6f}, {hf_result.max():.6f}]")
    print(f"  Is finite: {torch.isfinite(hf_result).all().item()}")
    print(f"  Non-zero fraction: {(hf_result != 0).float().mean():.4f}")
    print()

    # ========================================================================
    # Call actual FS2 method
    # ========================================================================
    print("=" * 80)
    print("FS2 _gaussian_topk (actual method call)")
    print("=" * 80)

    fs2_result = fs2_ffn._gaussian_topk(fs2_gate)
    print(f"  FS2 result range: [{fs2_result.min()}, {fs2_result.max()}]")
    print(f"  Is finite: {torch.isfinite(fs2_result).all().item()}")
    print(f"  Non-zero fraction: {(fs2_result != 0).float().mean():.4f}")
    print()

    if not torch.isfinite(fs2_result).all():
        print("❌ FS2 _gaussian_topk produces non-finite values!")
        print(f"  Inf count: {torch.isinf(fs2_result).sum().item()}")
        print(f"  NaN count: {torch.isnan(fs2_result).sum().item()}")
    else:
        print("✅ FS2 _gaussian_topk produces finite values")

"""Test 5: Verify Gaussian top-k implementations match."""

import torch
import torch.nn.functional as F

device = torch.device("cpu")
dtype = torch.float32

# Create test input
torch.manual_seed(42)
test_input = torch.randn(1, 4, 8192, device=device, dtype=dtype)
activation_sparsity = 0.95

print("=" * 80)
print("TEST 5: Compare Gaussian Top-K Implementations")
print("=" * 80)
print(f"Input shape: {test_input.shape}")
print(f"Activation sparsity: {activation_sparsity}")
print()

# ========================================================================
# HF implementation
# ========================================================================
print("HF Gaussian Top-K:")
print("-" * 80)

target_sparsity_tensor = torch.tensor(activation_sparsity, dtype=torch.float32, device=device)
normal_dist = torch.distributions.normal.Normal(0, 1)
std_multiplier_hf = normal_dist.icdf(target_sparsity_tensor)
std_multiplier_hf = std_multiplier_hf.type(dtype)

inputs_mean_hf = torch.mean(test_input, dim=-1, keepdim=True)
inputs_std_hf = torch.std(test_input, dim=-1, keepdim=True, unbiased=False)
cutoff_hf = inputs_mean_hf + inputs_std_hf * std_multiplier_hf
hf_output = F.relu(test_input - cutoff_hf)

print(f"  std_multiplier: {std_multiplier_hf.item()}")
print(f"  mean: {inputs_mean_hf[0, 0, 0].item():.6f}")
print(f"  std: {inputs_std_hf[0, 0, 0].item():.6f}")
print(f"  cutoff: {cutoff_hf[0, 0, 0].item():.6f}")
print(f"  Output range: [{hf_output.min():.4f}, {hf_output.max():.4f}]")
print(f"  Non-zero fraction: {(hf_output != 0).float().mean():.4f}")
print()

# ========================================================================
# FS2 implementation
# ========================================================================
print("FS2 Gaussian Top-K:")
print("-" * 80)

target_sparsity = torch.tensor(activation_sparsity, dtype=torch.float32, device=device)
normal_dist_fs2 = torch.distributions.normal.Normal(0, 1)
std_multiplier_fs2 = normal_dist_fs2.icdf(target_sparsity).to(dtype)

inputs_mean_fs2 = torch.mean(test_input, dim=-1, keepdim=True)
inputs_std_fs2 = torch.std(test_input, dim=-1, keepdim=True, unbiased=False)
cutoff_fs2 = inputs_mean_fs2 + inputs_std_fs2 * std_multiplier_fs2
fs2_output = F.relu(test_input - cutoff_fs2)

print(f"  std_multiplier: {std_multiplier_fs2.item()}")
print(f"  mean: {inputs_mean_fs2[0, 0, 0].item():.6f}")
print(f"  std: {inputs_std_fs2[0, 0, 0].item():.6f}")
print(f"  cutoff: {cutoff_fs2[0, 0, 0].item():.6f}")
print(f"  Output range: [{fs2_output.min():.4f}, {fs2_output.max():.4f}]")
print(f"  Non-zero fraction: {(fs2_output != 0).float().mean():.4f}")
print()

# ========================================================================
# Comparison
# ========================================================================
print("=" * 80)
print("COMPARISON")
print("=" * 80)

diff = (hf_output - fs2_output).abs()
print(f"Max diff: {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")

if diff.max() < 1e-10:
    print("✅ Implementations are identical")
else:
    print(f"❌ Implementations differ!")
    print(f"\nHF output sample [0, 0, :5]: {hf_output[0, 0, :5]}")
    print(f"FS2 output sample [0, 0, :5]: {fs2_output[0, 0, :5]}")

#!/usr/bin/env python3
"""Compare Q,K,V at layer 20 when using shared KV vs computing own KV."""

import torch
from transformers import AutoModelForCausalLM
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.models.gemma3n.kv_projection import KVProjectionType
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
config = get_gemma3n_e2b_config()

# Load HF checkpoint once
print("Loading HF checkpoint...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
)
fs2_state_dict = convert_gemma3n_state_dict(hf_model.state_dict(), config)

# Test 1: WITH KV sharing (consumer uses shared K/V)
print("\n" + "=" * 80)
print("TEST 1: Layer 20 WITH KV sharing")
print("=" * 80)

model1 = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
model1.load_state_dict(fs2_state_dict, strict=False)

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
batch_layout = BatchLayout(input_ids.shape, [5], device=device)

kv_slots = {KVProjectionType.LOCAL: None, KVProjectionType.GLOBAL: None}

# Monkey-patch SDPA to capture Q, K, V
captured_qkv_shared = {}

original_sdpa = model1.decoder.layers[20].self_attn.sdpa.forward

def capture_sdpa_shared(q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
    captured_qkv_shared['q'] = q.clone()
    captured_qkv_shared['k'] = k.clone()
    captured_qkv_shared['v'] = v.clone()
    return original_sdpa(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)

model1.decoder.layers[20].self_attn.sdpa.forward = capture_sdpa_shared

with torch.no_grad():
    embeds, _ = model1.decoder_frontend(input_ids, batch_layout, state_bag=None)
    hidden = model1.decoder._stack_altup(embeds)
    attn_bias_cache = AttentionBiasCache()

    for idx in range(21):
        hidden = model1.decoder.layers[idx](
            hidden, batch_layout, attn_bias_cache,
            per_layer_input=None, state_bag=None,
            kv_projection_slots=kv_slots
        )

print(f"Layer 20 WITH sharing:")
print(f"  Q shape: {captured_qkv_shared['q'].shape}")
print(f"  K shape: {captured_qkv_shared['k'].shape}")
print(f"  V shape: {captured_qkv_shared['v'].shape}")

# Test 2: WITHOUT KV sharing (compute own K/V)
print("\n" + "=" * 80)
print("TEST 2: Layer 20 WITHOUT KV sharing (computes own K/V)")
print("=" * 80)

model2 = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
model2.load_state_dict(fs2_state_dict, strict=False)  # Same checkpoint!

# Don't pass kv_projection_slots - forces layer 20 to compute own K/V
captured_qkv_own = {}

original_sdpa2 = model2.decoder.layers[20].self_attn.sdpa.forward

def capture_sdpa_own(q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
    captured_qkv_own['q'] = q.clone()
    captured_qkv_own['k'] = k.clone()
    captured_qkv_own['v'] = v.clone()
    return original_sdpa2(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)

model2.decoder.layers[20].self_attn.sdpa.forward = capture_sdpa_own

with torch.no_grad():
    embeds2, _ = model2.decoder_frontend(input_ids, batch_layout, state_bag=None)
    hidden2 = model2.decoder._stack_altup(embeds2)
    attn_bias_cache2 = AttentionBiasCache()

    for idx in range(21):
        hidden2 = model2.decoder.layers[idx](
            hidden2, batch_layout, attn_bias_cache2,
            per_layer_input=None, state_bag=None,
            kv_projection_slots=None  # Force compute own K/V
        )

print(f"Layer 20 WITHOUT sharing (own K/V):")
print(f"  Q shape: {captured_qkv_own['q'].shape}")
print(f"  K shape: {captured_qkv_own['k'].shape}")
print(f"  V shape: {captured_qkv_own['v'].shape}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

q_diff = (captured_qkv_shared['q'] - captured_qkv_own['q']).abs()
k_diff = (captured_qkv_shared['k'] - captured_qkv_own['k']).abs()
v_diff = (captured_qkv_shared['v'] - captured_qkv_own['v']).abs()

print(f"\nQ difference (should be ~0, both compute own Q):")
print(f"  Max:  {q_diff.max().item():.6e}")

print(f"\nK difference (shared from layer 18 vs computed fresh):")
print(f"  Max:  {k_diff.max().item():.6e}")
print(f"  Mean: {k_diff.mean().item():.6e}")

print(f"\nV difference (shared from layer 18 vs computed fresh):")
print(f"  Max:  {v_diff.max().item():.6e}")
print(f"  Mean: {v_diff.mean().item():.6e}")

print("\n" + "=" * 80)
if q_diff.max() > 1e-4:
    print("❌ Q differs! Both runs should compute same Q from same inputs.")
elif k_diff.max() > 1e-2:
    print("❌ Shared K/V differs SIGNIFICANTLY from freshly computed K/V!")
    print("   Layer 18's K/V (from layer 18's input) != Layer 20's K/V (from layer 20's input)")
    print("   This is EXPECTED if HF wants consumers to use source's K/V computed from different hidden states.")
else:
    print("✅ Shared K/V matches freshly computed K/V (unexpected!)")


#!/usr/bin/env python3
"""
Test: Disable HF's use_cache to see if parity improves.
If HF KV sharing is the issue, disabling it should give us parity.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

print("="*80)
print("TEST: Disable HF KV Cache to Check if Sharing is the Issue")
print("="*80)

# Load models
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)

text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print(f"\nInput: \"{text}\"")
print(f"Shape: {input_ids.shape}")

with torch.no_grad():
    # Test 1: HF WITH cache (default behavior)
    print(f"\n{'='*80}")
    print("TEST 1: HF with use_cache=True (default, KV sharing ACTIVE)")
    print(f"{'='*80}")

    hf_logits_cached = hf_model(input_ids, use_cache=True).logits
    print(f"HF (cached) logits[0, -1, :5]: {hf_logits_cached[0, -1, :5]}")

    # Test 2: HF WITHOUT cache (disable KV sharing)
    print(f"\n{'='*80}")
    print("TEST 2: HF with use_cache=False (KV sharing DISABLED)")
    print(f"{'='*80}")

    hf_logits_no_cache = hf_model(input_ids, use_cache=False).logits
    print(f"HF (no cache) logits[0, -1, :5]: {hf_logits_no_cache[0, -1, :5]}")

    # Test 3: FS2 (no KV sharing implemented)
    print(f"\n{'='*80}")
    print("TEST 3: FS2 (no KV sharing)")
    print(f"{'='*80}")

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    fs2_logits = fs2_model(input_ids, batch_layout)
    print(f"FS2 logits[0, -1, :5]: {fs2_logits[0, -1, :5]}")

    # Comparisons
    print(f"\n{'='*80}")
    print("COMPARISONS")
    print(f"{'='*80}")

    # HF cached vs HF no-cache
    diff_hf = (hf_logits_cached - hf_logits_no_cache).abs()
    print(f"\nHF (cached) vs HF (no cache):")
    print(f"  Max diff:  {diff_hf.max().item():.6e}")
    print(f"  Mean diff: {diff_hf.mean().item():.6e}")

    if diff_hf.max().item() < 1e-6:
        print(f"  ✅ Identical (KV sharing has no effect)")
    else:
        print(f"  ❌ DIFFERENT (KV sharing changes outputs!)")

    # HF cached vs FS2
    diff_cached = (hf_logits_cached - fs2_logits).abs()
    print(f"\nHF (cached) vs FS2:")
    print(f"  Max diff:  {diff_cached.max().item():.6e}")
    print(f"  Mean diff: {diff_cached.mean().item():.6e}")

    # HF no-cache vs FS2
    diff_no_cache = (hf_logits_no_cache - fs2_logits).abs()
    print(f"\nHF (no cache) vs FS2:")
    print(f"  Max diff:  {diff_no_cache.max().item():.6e}")
    print(f"  Mean diff: {diff_no_cache.mean().item():.6e}")

    if diff_no_cache.max().item() < 1e-3:
        print(f"  ✅ PARITY ACHIEVED!")
        print(f"  ROOT CAUSE CONFIRMED: Missing KV sharing in FS2")
    else:
        print(f"  ❌ Still diverges")
        print(f"  KV sharing is NOT the only issue")

print(f"\n{'='*80}")

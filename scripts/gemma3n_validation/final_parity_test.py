#!/usr/bin/env python3
"""
Final comprehensive parity test: Full 30-layer forward pass comparison.
Tests that HF and FS2 produce identical outputs after scaling fix.
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
print("FINAL COMPREHENSIVE PARITY TEST")
print("="*80)

# Load models
print("\n[Loading models...]")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)

# Test cases
test_cases = [
    "The quick brown",
    "Hello, how are you?",
    "In a galaxy far, far away",
]

print("\n[Running tests...]")

all_passed = True

for i, text in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: \"{text}\"")
    print(f"{'='*80}")

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        # HF forward
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

        # FS2 forward (returns logits directly as tensor)
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        fs2_logits = fs2_model(input_ids, batch_layout)

        # Compare
        diff = (hf_logits - fs2_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\nLogits comparison:")
        print(f"  HF shape:  {hf_logits.shape}")
        print(f"  FS2 shape: {fs2_logits.shape}")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        # Sample values
        print(f"\n  HF logits [0, -1, :5]:  {hf_logits[0, -1, :5]}")
        print(f"  FS2 logits [0, -1, :5]: {fs2_logits[0, -1, :5]}")

        # Check predictions match
        hf_pred = hf_logits[0, -1].argmax().item()
        fs2_pred = fs2_logits[0, -1].argmax().item()

        print(f"\n  HF prediction:  {hf_pred} ({tokenizer.decode([hf_pred])})")
        print(f"  FS2 prediction: {fs2_pred} ({tokenizer.decode([fs2_pred])})")

        # Pass/fail
        THRESHOLD = 1e-4  # Allow slightly higher threshold for 30-layer accumulation
        if max_diff < THRESHOLD:
            print(f"\n  ✅ PASS (max diff < {THRESHOLD})")
        else:
            print(f"\n  ❌ FAIL (max diff >= {THRESHOLD})")
            all_passed = False

        if hf_pred == fs2_pred:
            print(f"  ✅ Predictions match")
        else:
            print(f"  ⚠️  Predictions differ")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if all_passed:
    print("✅ ALL TESTS PASSED! Full model parity achieved.")
else:
    print("❌ Some tests failed. Check details above.")

print(f"{'='*80}")

#!/usr/bin/env python3
"""
Simple full model parity test after scaling fix.
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
print("FULL MODEL PARITY TEST (Post Scaling Fix)")
print("="*80)

# Load models
print("\nLoading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)

# Test with simple input
text = "The quick brown"
print(f"\nInput: \"{text}\"")

input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
print(f"Input shape: {input_ids.shape}")

with torch.no_grad():
    # HF forward
    print("\nRunning HF model...")
    hf_outputs = hf_model(input_ids)
    hf_logits = hf_outputs.logits

    # FS2 forward
    print("Running FS2 model...")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    fs2_logits = fs2_model(input_ids, batch_layout)

    # Compare
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nLogits shapes:")
    print(f"  HF:  {hf_logits.shape}")
    print(f"  FS2: {fs2_logits.shape}")

    diff = (hf_logits - fs2_logits).abs()
    print(f"\nLogits difference:")
    print(f"  Max:  {diff.max().item():.6e}")
    print(f"  Mean: {diff.mean().item():.6e}")

    # Sample values
    print(f"\nSample logits at position [0, -1, :5]:")
    print(f"  HF:  {hf_logits[0, -1, :5]}")
    print(f"  FS2: {fs2_logits[0, -1, :5]}")

    # Predictions
    hf_pred = hf_logits[0, -1].argmax().item()
    fs2_pred = fs2_logits[0, -1].argmax().item()

    hf_token = tokenizer.decode([hf_pred])
    fs2_token = tokenizer.decode([fs2_pred])

    print(f"\nNext token prediction:")
    print(f"  HF:  {hf_pred} -> '{hf_token}'")
    print(f"  FS2: {fs2_pred} -> '{fs2_token}'")

    # Overall assessment
    print(f"\n" + "="*80)
    THRESHOLD = 1e-3  # Be generous for 30-layer accumulation

    if diff.max().item() < THRESHOLD:
        print(f"✅ PASS: Max diff {diff.max().item():.6e} < {THRESHOLD}")
    else:
        print(f"❌ FAIL: Max diff {diff.max().item():.6e} >= {THRESHOLD}")
        print(f"\nThis suggests the scaling fix didn't fully resolve the issue.")
        print(f"Possible causes:")
        print(f"  - Fix not applied correctly")
        print(f"  - Additional divergence in other components")
        print(f"  - Accumulated error over 30 layers")

    if hf_pred == fs2_pred:
        print(f"✅ Predictions match")
    else:
        print(f"⚠️  Predictions differ (HF: '{hf_token}' vs FS2: '{fs2_token}')")

    print("="*80)

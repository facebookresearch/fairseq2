#!/usr/bin/env python3
"""
Position-dependent divergence test.
Check if parity holds for first token but fails for subsequent tokens.
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
print("POSITION-DEPENDENT DIVERGENCE TEST")
print("="*80)

# Load models
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)

text = "The quick brown fox"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print(f"\nInput: \"{text}\"")
print(f"Tokenized shape: {input_ids.shape}")
print(f"Tokens: {input_ids[0].tolist()}")

with torch.no_grad():
    # HF
    hf_logits = hf_model(input_ids).logits

    # FS2
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    fs2_logits = fs2_model(input_ids, batch_layout)

    print(f"\n{'='*80}")
    print("PER-POSITION ANALYSIS")
    print(f"{'='*80}")

    for pos in range(input_ids.shape[1]):
        hf_pos = hf_logits[0, pos]
        fs2_pos = fs2_logits[0, pos]

        diff = (hf_pos - fs2_pos).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        hf_pred = hf_pos.argmax().item()
        fs2_pred = fs2_pos.argmax().item()

        match = "✅" if hf_pred == fs2_pred else "❌"

        print(f"\nPosition {pos}:")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  HF pred:   {hf_pred} ({tokenizer.decode([hf_pred])})")
        print(f"  FS2 pred:  {fs2_pred} ({tokenizer.decode([fs2_pred])})")
        print(f"  Match: {match}")

        if max_diff < 1e-3:
            print(f"  Status: ✅ PARITY")
        elif max_diff < 1.0:
            print(f"  Status: ⚠️  SMALL DIVERGENCE")
        else:
            print(f"  Status: ❌ LARGE DIVERGENCE")

print(f"\n{'='*80}")

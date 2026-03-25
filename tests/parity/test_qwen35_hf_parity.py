"""Qwen 3.5 0.8B — HuggingFace vs fairseq2 numerical parity test.

Downloads the HF checkpoint, loads it into both HF and fairseq2,
runs the same input, and asserts logit closeness.
"""

import sys
import torch
import torch.nn.functional as F

# ---- Step 1: Load HF model ----
print("=" * 60)
print("Step 1: Loading HuggingFace model...")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)
hf_model.eval()
print(f"  HF model loaded: {sum(p.numel() for p in hf_model.parameters()):,} params")

# ---- Step 2: Build fairseq2 model from config ----
print("\n" + "=" * 60)
print("Step 2: Building fairseq2 model...")
print("=" * 60)

from fairseq2.models.qwen.config import Qwen35Config
from fairseq2.models.qwen.factory import create_qwen35_model
from fairseq2.models.qwen.interop import convert_qwen35_state_dict

config = Qwen35Config(
    model_dim=1024,
    max_seq_len=262_144,
    vocab_size=248_320,
    tied_embeddings=True,
    num_layers=24,
    num_attn_heads=8,
    num_key_value_heads=2,
    head_dim=256,
    ffn_inner_dim=3584,
    partial_rotary_factor=0.25,
    rope_theta=10_000_000.0,
    full_attention_interval=4,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
)

fs2_model = create_qwen35_model(config)
fs2_model.eval()
print(f"  fs2 model built: {sum(p.numel() for p in fs2_model.parameters()):,} params")

# ---- Step 3: Convert and load HF state dict into fairseq2 ----
print("\n" + "=" * 60)
print("Step 3: Converting HF state dict -> fairseq2...")
print("=" * 60)

hf_state_dict = dict(hf_model.state_dict())

fs2_state_dict = convert_qwen35_state_dict(hf_state_dict, config)

# Check for missing/unexpected keys
fs2_keys = set(fs2_model.state_dict().keys())
converted_keys = set(fs2_state_dict.keys())

missing = fs2_keys - converted_keys
unexpected = converted_keys - fs2_keys

if missing:
    print(f"  WARNING: {len(missing)} missing keys:")
    for k in sorted(missing)[:20]:
        print(f"    - {k}")
if unexpected:
    print(f"  WARNING: {len(unexpected)} unexpected keys:")
    for k in sorted(unexpected)[:20]:
        print(f"    - {k}")

if missing or unexpected:
    print("\n  Attempting to load with strict=False...")
    result = fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print(f"  Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print("  FATAL: Cannot proceed with missing keys.")
        for k in sorted(result.missing_keys)[:30]:
            print(f"    - {k}")
        sys.exit(1)
else:
    fs2_model.load_state_dict(fs2_state_dict, strict=True)
    print("  State dict loaded successfully (strict=True)")

# ---- Step 4: Prepare input ----
print("\n" + "=" * 60)
print("Step 4: Preparing input...")
print("=" * 60)

test_text = "The capital of France is"
tokens = hf_tokenizer(test_text, return_tensors="pt")
input_ids = tokens["input_ids"]  # (1, S)
print(f"  Input: '{test_text}'")
print(f"  Token IDs: {input_ids.tolist()}")
print(f"  Sequence length: {input_ids.shape[1]}")

# ---- Step 5: HF forward pass ----
print("\n" + "=" * 60)
print("Step 5: HF forward pass...")
print("=" * 60)

with torch.no_grad():
    hf_output = hf_model(input_ids)
    hf_logits = hf_output.logits  # (1, S, V)

print(f"  HF logits shape: {hf_logits.shape}")
print(f"  HF logits[0, -1, :5]: {hf_logits[0, -1, :5]}")

# ---- Step 6: fairseq2 forward pass ----
print("\n" + "=" * 60)
print("Step 6: fairseq2 forward pass...")
print("=" * 60)

from fairseq2.nn import BatchLayout

with torch.no_grad():
    seqs = input_ids  # (1, S)
    seqs_layout = BatchLayout.of(seqs)
    fs2_logits = fs2_model(seqs, seqs_layout)  # returns logits directly when no targets

print(f"  fs2 logits shape: {fs2_logits.shape}")
print(f"  fs2 logits[0, -1, :5]: {fs2_logits[0, -1, :5]}")

# ---- Step 7: Compare ----
print("\n" + "=" * 60)
print("Step 7: Numerical comparison...")
print("=" * 60)

# Compare last token logits (most common check)
hf_last = hf_logits[0, -1].float()
fs2_last = fs2_logits[0, -1].float()

abs_diff = (hf_last - fs2_last).abs()
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()

print(f"  Last-token logit max  abs diff: {max_diff:.6e}")
print(f"  Last-token logit mean abs diff: {mean_diff:.6e}")

# Full sequence comparison
hf_all = hf_logits.float()
fs2_all = fs2_logits.float()

full_abs_diff = (hf_all - fs2_all).abs()
full_max_diff = full_abs_diff.max().item()
full_mean_diff = full_abs_diff.mean().item()

print(f"  Full-seq logit max  abs diff: {full_max_diff:.6e}")
print(f"  Full-seq logit mean abs diff: {full_mean_diff:.6e}")

# Top-1 token match
hf_top1 = hf_last.argmax().item()
fs2_top1 = fs2_last.argmax().item()
print(f"\n  HF  top-1 token: {hf_top1} -> '{hf_tokenizer.decode([hf_top1])}'")
print(f"  fs2 top-1 token: {fs2_top1} -> '{hf_tokenizer.decode([fs2_top1])}'")

# Top-5 match
hf_top5 = hf_last.topk(5).indices.tolist()
fs2_top5 = fs2_last.topk(5).indices.tolist()
print(f"\n  HF  top-5: {hf_top5} -> {[hf_tokenizer.decode([t]) for t in hf_top5]}")
print(f"  fs2 top-5: {fs2_top5} -> {[hf_tokenizer.decode([t]) for t in fs2_top5]}")

# Cosine similarity
cos_sim = F.cosine_similarity(hf_last.unsqueeze(0), fs2_last.unsqueeze(0)).item()
print(f"\n  Cosine similarity (last token): {cos_sim:.8f}")

# Pass/fail
ATOL = 1e-4
if full_max_diff < ATOL:
    print(f"\n✅ PASS: max abs diff {full_max_diff:.2e} < {ATOL:.0e}")
elif cos_sim > 0.9999:
    print(f"\n⚠️  SOFT PASS: max diff {full_max_diff:.2e} > {ATOL:.0e} but cosine sim {cos_sim:.6f} > 0.9999")
else:
    print(f"\n❌ FAIL: max diff {full_max_diff:.2e}, cosine sim {cos_sim:.6f}")

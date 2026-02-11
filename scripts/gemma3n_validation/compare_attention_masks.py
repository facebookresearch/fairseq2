#!/usr/bin/env python3
"""Compare attention masks between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("ATTENTION MASK/BIAS COMPARISON")
print("="*80)

text_config = hf_model.model.language_model.config
seq_len = input_ids.shape[1]

print(f"\nSequence length: {seq_len}")
print(f"Config sliding_window: {text_config.sliding_window}")

# Get layer 0 attention
hf_layer0 = hf_model.model.language_model.layers[0]
fs2_layer0 = fs2_model.decoder.layers[0]

print(f"Layer 0 attention type: {hf_layer0.attention_type}")
print(f"FS2 attention bias type: {type(fs2_layer0.self_attn.sdpa.bias)}")
print(f"FS2 sliding window: {fs2_layer0.self_attn.sdpa.bias.attn_window_len}")

# HF creates sliding window causal mask
# Let's create it manually using the same logic
print(f"\n[HF ATTENTION MASK]")

# Create causal mask
mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask.triu_(diagonal=1)  # Upper triangle is -inf (can't attend to future)

# Apply sliding window: mask positions more than window_len back
if text_config.sliding_window is not None and text_config.sliding_window < seq_len:
    # Can only attend to window_len positions back
    # So for position i, can attend to [max(0, i - window_len + 1), i]
    mask.triu_(diagonal=1 - text_config.sliding_window)  # Lower triangle beyond window is -inf

hf_mask = mask
print(f"HF mask shape: {hf_mask.shape}")
print(f"HF mask dtype: {hf_mask.dtype}")
print(f"HF mask sample [:10, :10]:\n{hf_mask[:10, :10]}")

# FS2 creates attention bias
print(f"\n[FS2 ATTENTION BIAS]")
seq_lens = [seq_len]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

# Create the bias tensor directly using the bias object
fs2_bias_obj = fs2_layer0.self_attn.sdpa.bias
fs2_bias_tensor = fs2_bias_obj.create_bias_tensor(seq_len, seq_len, device=device, dtype=dtype)

print(f"FS2 bias shape: {fs2_bias_tensor.shape}")
print(f"FS2 bias dtype: {fs2_bias_tensor.dtype}")
print(f"FS2 bias sample [:10, :10]:\n{fs2_bias_tensor[:10, :10]}")

# Compare
print(f"\n[COMPARISON]")
# Both are now [seq, seq] with 0 for valid, -inf for masked

print(f"HF mask unique values: {torch.unique(hf_mask)}")
print(f"FS2 bias unique values: {torch.unique(fs2_bias_tensor)}")

diff = (hf_mask - fs2_bias_tensor).abs()
print(f"Mask diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

# Check pattern
print(f"\nHF mask pattern (0=valid, -inf=masked):")
print(f"  Row 0: {(hf_mask[0] == 0).sum().item()} valid positions")
print(f"  Row 5: {(hf_mask[5] == 0).sum().item()} valid positions")
print(f"  Row 9: {(hf_mask[9] == 0).sum().item()} valid positions")

print(f"\nFS2 bias pattern (0=valid, -inf=masked):")
print(f"  Row 0: {(fs2_bias_tensor[0] == 0).sum().item()} valid positions")
print(f"  Row 5: {(fs2_bias_tensor[5] == 0).sum().item()} valid positions")
print(f"  Row 9: {(fs2_bias_tensor[9] == 0).sum().item()} valid positions")

print("="*80)

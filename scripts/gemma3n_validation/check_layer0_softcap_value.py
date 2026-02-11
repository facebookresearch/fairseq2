#!/usr/bin/env python3
"""Check if HF layer0 self_attn has a softcap attribute/buffer."""

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

layer0 = hf_model.model.language_model.layers[0]

print("="*80)
print("CHECK LAYER 0 SOFTCAP ATTRIBUTE")
print("="*80)

# Check if softcap exists
if hasattr(layer0.self_attn, 'softcap'):
    print(f"\n✓ layer0.self_attn.softcap exists!")
    print(f"  Value: {layer0.self_attn.softcap}")
    print(f"  Type: {type(layer0.self_attn.softcap)}")
else:
    print(f"\n❌ layer0.self_attn.softcap does NOT exist")

# Check for attention_logits_soft_cap
if hasattr(layer0.self_attn, 'attention_logits_soft_cap'):
    print(f"\n✓ layer0.self_attn.attention_logits_soft_cap exists!")
    print(f"  Value: {layer0.self_attn.attention_logits_soft_cap}")
else:
    print(f"\n❌ layer0.self_attn.attention_logits_soft_cap does NOT exist")

# Check named buffers
print(f"\n[Named buffers in layer0.self_attn]:")
for name, buffer in layer0.self_attn.named_buffers():
    print(f"  {name}: {buffer.shape if hasattr(buffer, 'shape') else buffer}")

# Check config for text attention softcap
print(f"\n[Text config check]:")
text_config = hf_model.model.language_model.config
attrs = ['conf_attention_logit_cap', 'text_attention_logit_cap', 'attn_logit_softcapping']
for attr in attrs:
    if hasattr(text_config, attr):
        val = getattr(text_config, attr)
        print(f"  {attr}: {val}")

print("="*80)

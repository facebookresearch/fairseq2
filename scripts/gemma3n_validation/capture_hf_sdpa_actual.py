#!/usr/bin/env python3
"""Capture HF's actual SDPA call by hooking at a lower level."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("CAPTURE HF ACTUAL SDPA CALL")
print("="*80)

# Patch at the module level where HF imports it
import sys
import torch.nn.functional as F

# Save original
_original_sdpa = F.scaled_dot_product_attention

call_log = []

def sdpa_capture(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    call_info = {
        'q_shape': query.shape,
        'k_shape': key.shape,
        'v_shape': value.shape,
        'is_causal': is_causal,
        'has_mask': attn_mask is not None,
    }
    call_log.append(call_info)

    print(f"\n[SDPA CALL #{len(call_log)}]")
    print(f"  Q: {query.shape}")
    print(f"  K: {key.shape}")
    print(f"  V: {value.shape}")
    print(f"  is_causal: {is_causal}")
    print(f"  mask: {attn_mask.shape if attn_mask is not None else None}")

    # Actually call it to see if it fails
    try:
        result = _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        print(f"  ✓ Call succeeded, output: {result.shape}")
        return result
    except Exception as e:
        print(f"  ✗ Call failed: {e}")
        raise

# Patch it everywhere
F.scaled_dot_product_attention = sdpa_capture
torch.nn.functional.scaled_dot_product_attention = sdpa_capture

# Also patch in transformers module if imported
for module_name, module in list(sys.modules.items()):
    if 'transformers' in module_name and hasattr(module, 'scaled_dot_product_attention'):
        setattr(module, 'scaled_dot_product_attention', sdpa_capture)

with torch.no_grad():
    text_config = hf_model.model.language_model.config
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    hf_layer0 = hf_lm.layers[0]
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    print(f"\n{'='*80}")
    print("RUNNING HF LAYER 0 ATTENTION")
    print(f"{'='*80}")

    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )

print(f"\n{'='*80}")
print(f"SUMMARY: {len(call_log)} SDPA calls captured")
print(f"{'='*80}")

if call_log:
    print(f"\nFirst call details:")
    for k, v in call_log[0].items():
        print(f"  {k}: {v}")

print("="*80)

"""Test what past_key_values is when use_cache=False vs True."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")

input_ids = tokenizer("Hello", return_tensors="pt").input_ids

print("="*80)
print("TEST: What is past_key_values when use_cache=False vs True?")
print("="*80)

# Hook to intercept layer 15 (first shared layer)
captured = {}

original_forward = model.model.language_model.layers[15].self_attn.forward

def hook_forward(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    captured['past_key_values'] = past_key_values
    captured['past_key_values_type'] = type(past_key_values).__name__ if past_key_values is not None else 'None'
    captured['is_kv_shared_layer'] = model.model.language_model.layers[15].self_attn.is_kv_shared_layer

    # Check if shared_layers exists
    if past_key_values is not None and hasattr(past_key_values, 'shared_layers'):
        captured['shared_layers_exists'] = True
        captured['shared_layers_keys'] = list(past_key_values.shared_layers.keys()) if past_key_values.shared_layers else []
    else:
        captured['shared_layers_exists'] = False
        captured['shared_layers_keys'] = []

    return original_forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

model.model.language_model.layers[15].self_attn.forward = hook_forward

print("\n1. Running with use_cache=False")
print("-"*80)
with torch.no_grad():
    output = model(input_ids, use_cache=False)

print(f"past_key_values type: {captured['past_key_values_type']}")
print(f"is_kv_shared_layer: {captured['is_kv_shared_layer']}")
print(f"shared_layers exists: {captured['shared_layers_exists']}")
print(f"shared_layers keys: {captured['shared_layers_keys']}")

if captured['past_key_values'] is None:
    print("\n⚠️  CRITICAL: past_key_values=None, so KV sharing is DISABLED!")
    print("   Line 1314: if self.is_kv_shared_layer and past_key_values is not None:")
    print("              This evaluates to FALSE, so shared layer computes its own K/V")

print("\n2. Running with use_cache=True")
print("-"*80)
captured.clear()
with torch.no_grad():
    output = model(input_ids, use_cache=True)

print(f"past_key_values type: {captured['past_key_values_type']}")
print(f"is_kv_shared_layer: {captured['is_kv_shared_layer']}")
print(f"shared_layers exists: {captured['shared_layers_exists']}")
print(f"shared_layers keys: {captured['shared_layers_keys']}")

if captured['past_key_values'] is not None:
    print("\n✓ past_key_values exists, KV sharing is ENABLED")
    print(f"  Source layers that stored K/V: {captured['shared_layers_keys']}")

# Restore
model.model.language_model.layers[15].self_attn.forward = original_forward

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("use_cache=False → past_key_values=None → KV sharing DISABLED")
print("use_cache=True  → past_key_values=Cache → KV sharing ENABLED")
print()
print("Our implementation ALWAYS passes shared_kv_cache, so we ALWAYS enable sharing.")
print("This is why we get parity with use_cache=False but not use_cache=True!")

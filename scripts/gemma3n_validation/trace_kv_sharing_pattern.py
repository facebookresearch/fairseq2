"""Trace HF KV sharing pattern to understand source mapping."""

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it")
config = model.model.language_model.config

print("HF KV Sharing Pattern")
print("="*80)

first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
prev_layers = config.layer_types[:first_kv_shared_layer_idx]

print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"num_kv_shared_layers: {config.num_kv_shared_layers}")
print(f"first_kv_shared_layer_idx: {first_kv_shared_layer_idx}")
print(f"\nLayer types for layers 0-{first_kv_shared_layer_idx-1} (prev_layers):")
print(prev_layers)
print()

print("Shared layer → source mapping:")
print("-"*80)

sources_used = {}

for layer_idx in range(config.num_hidden_layers):
    layer = model.model.language_model.layers[layer_idx]

    if layer.self_attn.is_kv_shared_layer:
        source_idx = layer.self_attn.kv_shared_layer_index
        layer_type = config.layer_types[layer_idx]

        if source_idx not in sources_used:
            sources_used[source_idx] = []
        sources_used[source_idx].append(layer_idx)

        print(f"Layer {layer_idx:2d} ({layer_type:6s}) → uses K/V from layer {source_idx:2d}")

print()
print("="*80)
print("SOURCE REUSE PATTERN:")
print("="*80)

for source_idx in sorted(sources_used.keys()):
    consumers = sources_used[source_idx]
    source_type = config.layer_types[source_idx]
    print(f"\nLayer {source_idx} ({source_type}) stores K/V ONCE")
    print(f"  → Reused by {len(consumers)} shared layers: {consumers}")

print()
print("KEY INSIGHT:")
print("-"*80)
print("Each source layer (13, 14) stores K/V ONCE in past_key_values.shared_layers")
print("ALL shared layers of the same type retrieve from the SAME source")
print("This is NOT iterative creation/consumption - it's simple reuse!")

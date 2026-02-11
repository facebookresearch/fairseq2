"""Check if HF shared layers have k_proj/v_proj weights."""

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it")

print("Checking which layers have k_proj/v_proj:")
print("="*80)

for i, layer in enumerate(model.model.language_model.layers):
    has_k_proj = hasattr(layer.self_attn, 'k_proj')
    has_v_proj = hasattr(layer.self_attn, 'v_proj')
    is_shared = layer.self_attn.is_kv_shared_layer if hasattr(layer.self_attn, 'is_kv_shared_layer') else False
    is_source = layer.self_attn.store_full_length_kv if hasattr(layer.self_attn, 'store_full_length_kv') else False

    status = []
    if is_source:
        status.append("SOURCE")
    if is_shared:
        source_idx = layer.self_attn.kv_shared_layer_index if hasattr(layer.self_attn, 'kv_shared_layer_index') else "?"
        status.append(f"SHARED from L{source_idx}")

    kv_proj_status = f"k_proj={has_k_proj}, v_proj={has_v_proj}"

    if status:
        print(f"Layer {i:2d}: {' | '.join(status):30s} - {kv_proj_status}")
    elif not has_k_proj or not has_v_proj:
        print(f"Layer {i:2d}: {'NORMAL':30s} - {kv_proj_status} ⚠️")

print()
print("Key observations:")
print("- If shared layers DON'T have k_proj/v_proj, FS2 needs to not create them")
print("- If shared layers DO have k_proj/v_proj but don't use them, FS2 is correct")

"""Check which layers have k_proj/v_proj in the checkpoint."""

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it")
state_dict = model.state_dict()

print("Checking checkpoint keys for k_proj/v_proj:")
print("="*80)

has_k_proj = {}
has_v_proj = {}

for key in state_dict.keys():
    if 'self_attn.k_proj' in key:
        # Extract layer number
        import re
        match = re.search(r'layers\.(\d+)\.self_attn\.k_proj', key)
        if match:
            layer_idx = int(match.group(1))
            has_k_proj[layer_idx] = True

    if 'self_attn.v_proj' in key:
        import re
        match = re.search(r'layers\.(\d+)\.self_attn\.v_proj', key)
        if match:
            layer_idx = int(match.group(1))
            has_v_proj[layer_idx] = True

print(f"\nLayers with k_proj weights: {sorted(has_k_proj.keys())}")
print(f"Layers with v_proj weights: {sorted(has_v_proj.keys())}")
print()

# Check layer configuration
for i in range(30):
    has_k = i in has_k_proj
    has_v = i in has_v_proj
    layer = model.model.language_model.layers[i]
    is_shared = layer.self_attn.is_kv_shared_layer if hasattr(layer.self_attn, 'is_kv_shared_layer') else False
    is_source = layer.self_attn.store_full_length_kv if hasattr(layer.self_attn, 'store_full_length_kv') else False

    status = "NORMAL"
    if is_source:
        status = "SOURCE"
    elif is_shared:
        source_idx = layer.self_attn.kv_shared_layer_index
        status = f"SHARED(L{source_idx})"

    checkpoint_status = f"k={has_k}, v={has_v}"

    if is_shared and (has_k or has_v):
        print(f"Layer {i:2d}: {status:15s} - checkpoint: {checkpoint_status} ⚠️ UNEXPECTED!")
    elif is_shared:
        print(f"Layer {i:2d}: {status:15s} - checkpoint: {checkpoint_status}")

print()
print("CRITICAL: If shared layers (15-29) have NO k_proj/v_proj in checkpoint,")
print("         then FS2 should NOT create these projections for shared layers!")

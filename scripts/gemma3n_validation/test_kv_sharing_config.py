"""Test KV sharing configuration logic."""

from fairseq2.models.gemma3n.config import get_kv_sharing_config, is_global_layer, get_gemma3n_e2b_config

print("=" * 80)
print("KV SHARING CONFIGURATION TEST")
print("=" * 80)

config = get_gemma3n_e2b_config()
num_layers = config.num_layers
num_kv_shared_layers = config.num_kv_shared_layers

print(f"\nConfig: {num_layers} layers, {num_kv_shared_layers} KV shared layers")
print(f"Expected: Layers 0-{num_layers - num_kv_shared_layers - 1} are source/normal")
print(f"Expected: Layers {num_layers - num_kv_shared_layers}-{num_layers - 1} share KV")
print()

# Track which layers are what type
source_layers = []
shared_layers = []
normal_layers = []

print("Layer Configuration:")
print("-" * 80)
print(f"{'Layer':>5} | {'Type':>8} | {'Global':>6} | {'KV Config':>20}")
print("-" * 80)

for layer_idx in range(num_layers):
    is_global = is_global_layer(layer_idx, num_layers)
    is_shared, source_idx, is_source = get_kv_sharing_config(
        layer_idx, num_layers, num_kv_shared_layers
    )

    # Categorize
    if is_shared:
        shared_layers.append(layer_idx)
        kv_config = f"Shared from L{source_idx}"
    elif is_source:
        source_layers.append(layer_idx)
        kv_config = "Source"
    else:
        normal_layers.append(layer_idx)
        kv_config = "Normal"

    layer_type = "Global" if is_global else "Local"

    print(f"{layer_idx:5d} | {layer_type:>8} | {'Yes' if is_global else 'No':>6} | {kv_config:>20}")

print("-" * 80)
print()

# Summary
print("Summary:")
print(f"  Normal layers (no KV sharing): {len(normal_layers)} - {normal_layers}")
print(f"  Source layers (store KV): {len(source_layers)} - {source_layers}")
print(f"  Shared layers (retrieve KV): {len(shared_layers)} - {shared_layers}")
print()

# Validation
print("Validation:")
errors = []

# Check count
if len(shared_layers) != num_kv_shared_layers:
    errors.append(f"Expected {num_kv_shared_layers} shared layers, got {len(shared_layers)}")

# Check that shared layers are the last N layers
expected_shared_start = num_layers - num_kv_shared_layers
if shared_layers and shared_layers[0] != expected_shared_start:
    errors.append(f"Expected shared layers to start at {expected_shared_start}, got {shared_layers[0]}")

# Check that each shared layer has a valid source
for shared_idx in shared_layers:
    is_shared, source_idx, is_source = get_kv_sharing_config(
        shared_idx, num_layers, num_kv_shared_layers
    )

    # Verify source exists and has same type
    if source_idx is None:
        errors.append(f"Shared layer {shared_idx} has no source")
    elif source_idx >= shared_idx:
        errors.append(f"Shared layer {shared_idx} source {source_idx} is not earlier")
    else:
        # Check same type
        shared_is_global = is_global_layer(shared_idx, num_layers)
        source_is_global = is_global_layer(source_idx, num_layers)
        if shared_is_global != source_is_global:
            errors.append(
                f"Shared layer {shared_idx} ({'global' if shared_is_global else 'local'}) "
                f"source {source_idx} is ({'global' if source_is_global else 'local'})"
            )

if errors:
    print("  ❌ ERRORS FOUND:")
    for error in errors:
        print(f"    - {error}")
else:
    print("  ✅ All validation checks passed")

print()
print("=" * 80)
print("KV SHARING MAPPING")
print("=" * 80)

# Group shared layers by their source
from collections import defaultdict
source_to_shared = defaultdict(list)

for shared_idx in shared_layers:
    _, source_idx, _ = get_kv_sharing_config(
        shared_idx, num_layers, num_kv_shared_layers
    )
    source_to_shared[source_idx].append(shared_idx)

print("\nSource → Shared mapping:")
for source_idx in sorted(source_to_shared.keys()):
    shared_list = source_to_shared[source_idx]
    is_global = "Global" if is_global_layer(source_idx, num_layers) else "Local"
    print(f"  Layer {source_idx:2d} ({is_global:6s}) → Layers {shared_list}")

print()

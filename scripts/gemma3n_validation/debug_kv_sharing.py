"""Debug: Check if KV sharing is actually being triggered."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")

print("\n" + "="*80)
print("CHECKING KV SHARING LAYER CONFIGURATION")
print("="*80)

# Check FS2 layer configuration
for i, layer in enumerate(fs2_model.decoder.layers):
    if layer.is_kv_shared_layer or layer.is_kv_source_layer:
        status = []
        if layer.is_kv_source_layer:
            status.append(f"SOURCE")
        if layer.is_kv_shared_layer:
            status.append(f"SHARED from L{layer.kv_source_layer_idx}")
        print(f"Layer {i:2d}: {' | '.join(status)}")

print("\n" + "="*80)
print("INSTRUMENTING REGISTRY TO TRACK CALLS")
print("="*80)

# Monkey-patch the registry to track calls
from fairseq2.models.gemma3n.kv_sharing import KVSharedLayerRegistry

original_store = KVSharedLayerRegistry.store_kv_for_sharing
original_retrieve = KVSharedLayerRegistry.retrieve_shared_kv

store_calls = []
retrieve_calls = []

def tracked_store(self, source_layer_idx, key_states, value_states):
    store_calls.append((source_layer_idx, key_states.shape, value_states.shape))
    return original_store(self, source_layer_idx, key_states, value_states)

def tracked_retrieve(self, consumer_layer_idx, source_layer_idx):
    result = original_retrieve(self, consumer_layer_idx, source_layer_idx)
    retrieve_calls.append((consumer_layer_idx, source_layer_idx, result[0].shape, result[1].shape))
    return result

KVSharedLayerRegistry.store_kv_for_sharing = tracked_store
KVSharedLayerRegistry.retrieve_shared_kv = tracked_retrieve

# Run inference
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
batch_layout = BatchLayout(input_ids.shape, [input_ids.shape[1]], device=device)

print("Running FS2 forward pass...")
with torch.no_grad():
    fs2_logits = fs2_model(input_ids, batch_layout)

print("\n" + "="*80)
print("KV SHARING ACTIVITY")
print("="*80)

print(f"\nStore calls: {len(store_calls)}")
for source_idx, k_shape, v_shape in store_calls:
    print(f"  Layer {source_idx} stored K={k_shape} V={v_shape}")

print(f"\nRetrieve calls: {len(retrieve_calls)}")
for consumer_idx, source_idx, k_shape, v_shape in retrieve_calls:
    print(f"  Layer {consumer_idx} retrieved from L{source_idx}: K={k_shape} V={v_shape}")

if len(store_calls) == 0:
    print("\n⚠️  WARNING: No store calls! KV sharing is not working.")
    print("   Likely cause: Callbacks not being invoked")

if len(retrieve_calls) == 0:
    print("\n⚠️  WARNING: No retrieve calls! KV sharing is not working.")
    print("   Likely cause: Shared layers not configured correctly")

expected_stores = 2  # L13 (local source) and L14 (global source)
expected_retrieves = 15  # 15 shared layers

if len(store_calls) != expected_stores:
    print(f"\n⚠️  WARNING: Expected {expected_stores} store calls, got {len(store_calls)}")

if len(retrieve_calls) != expected_retrieves:
    print(f"\n⚠️  WARNING: Expected {expected_retrieves} retrieve calls, got {len(retrieve_calls)}")

# Restore
KVSharedLayerRegistry.store_kv_for_sharing = original_store
KVSharedLayerRegistry.retrieve_shared_kv = original_retrieve

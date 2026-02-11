#!/usr/bin/env python3
"""Compare stored K/V tensors from SOURCE layers (13, 14) between HF and FS2."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.models.gemma3n.kv_projection import KVProjectionType
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout, IncrementalStateBag


device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(
    convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

print(f"Input: {text!r}")
print(f"Tokens: {input_ids.shape[1]}\n")

print("=" * 80)
print("COMPARING STORED K/V FROM SOURCE LAYERS")
print("=" * 80)

with torch.no_grad():
    # ===== HF FORWARD =====
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # Stack to 4D
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=dtype)
    temp_hidden_states = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    # Setup position embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_lm.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # HF cache
    hf_cache = DynamicCache()

    # Run through layers 0-14 with HF
    print("\n[1/2] Running HF layers 0-14...")
    hf_hidden = hf_hidden_4d
    hf_state_bag = IncrementalStateBag(input_ids.shape[1])
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, BatchLayout(input_ids.shape, [input_ids.shape[1]], device=device), state_bag=hf_state_bag)
    per_layer_inputs = hf_state_bag.per_layer_inputs

    for layer_idx in range(15):
        hf_layer = hf_lm.layers[layer_idx]
        layer_type = hf_layer.attention_type
        layer_ple = per_layer_inputs[:, :, layer_idx, :]

        hf_hidden = hf_layer(
            hf_hidden,
            position_embeddings=position_embeddings[layer_type],
            per_layer_input=layer_ple,
            past_key_value=hf_cache,
        )

    # Extract stored K/V from HF cache at layers 13 and 14
    print("  Extracting HF cached K/V from layers 13 and 14...")

    # DynamicCache stores layers as a list of layer cache objects
    print(f"  Cache type: {type(hf_cache)}")
    print(f"  Cache.layers: {hf_cache.layers}")
    print(f"  Cache.layers length: {len(hf_cache.layers)}")
    print(f"  Cache seq length: {hf_cache.get_seq_length()}")

    if len(hf_cache.layers) == 0:
        print("\n⚠️  HF cache is EMPTY after running 15 layers!")
        print("     This means HF layers are NOT using past_key_value.update()")
        print("     Need to check if Gemma3n uses a different caching mechanism.")

        # Check what the HF layer actually does with the cache
        print("\n  Inspecting HF layer 13 to see if it has kv_sharing config...")
        hf_layer13 = hf_lm.layers[13]
        print(f"    Layer 13 type: {type(hf_layer13)}")
        print(f"    Has 'is_kv_shared_layer': {hasattr(hf_layer13, 'is_kv_shared_layer')}")
        print(f"    Has 'is_kv_source_layer': {hasattr(hf_layer13, 'is_kv_source_layer')}")
        print(f"    Has 'kv_source_layer_idx': {hasattr(hf_layer13, 'kv_source_layer_idx')}")

        if hasattr(hf_layer13, 'is_kv_source_layer'):
            print(f"    is_kv_source_layer: {hf_layer13.is_kv_source_layer}")

        raise RuntimeError("HF cache is empty - need to understand HF caching mechanism")

    # Access layer 13 and 14 caches
    layer13_cache = hf_cache.layers[13]
    layer14_cache = hf_cache.layers[14]

    # Each layer cache should have K/V tensors
    print(f"  Layer 13 cache type: {type(layer13_cache)}")

    # Try to access K/V - check if it's a tuple or has attributes
    if isinstance(layer13_cache, tuple):
        hf_layer13_k, hf_layer13_v = layer13_cache
        hf_layer14_k, hf_layer14_v = layer14_cache
    elif hasattr(layer13_cache, 'key_states'):
        hf_layer13_k = layer13_cache.key_states
        hf_layer13_v = layer13_cache.value_states
        hf_layer14_k = layer14_cache.key_states
        hf_layer14_v = layer14_cache.value_states
    else:
        print(f"  Layer 13 cache content: {layer13_cache}")
        raise RuntimeError("Cannot determine how to access K/V from layer cache")

    print(f"  HF Layer 13 K shape: {hf_layer13_k.shape}")
    print(f"  HF Layer 13 V shape: {hf_layer13_v.shape}")
    print(f"  HF Layer 14 K shape: {hf_layer14_k.shape}")
    print(f"  HF Layer 14 V shape: {hf_layer14_v.shape}")

    # ===== FS2 FORWARD =====
    print("\n[2/2] Running FS2 layers 0-14...")

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(input_ids.shape[1])

    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(fs2_embeds)

    # FS2 KV projection slots
    fs2_kv_slots = {
        KVProjectionType.LOCAL: None,
        KVProjectionType.GLOBAL: None,
    }

    attn_bias_cache = AttentionBiasCache()
    per_layer_inputs = state_bag.per_layer_inputs

    fs2_hidden = fs2_hidden_4d
    for layer_idx in range(15):
        fs2_layer = fs2_model.decoder.layers[layer_idx]
        layer_ple = per_layer_inputs[:, :, layer_idx, :]

        fs2_hidden = fs2_layer(
            fs2_hidden,
            batch_layout,
            attn_bias_cache,
            per_layer_input=layer_ple,
            state_bag=state_bag,
            kv_projection_slots=fs2_kv_slots,
        )

    # Extract stored K/V from FS2 slots
    print("  Extracting FS2 cached K/V from slots...")
    fs2_layer13_k, fs2_layer13_v = fs2_kv_slots[KVProjectionType.LOCAL]
    fs2_layer14_k, fs2_layer14_v = fs2_kv_slots[KVProjectionType.GLOBAL]

    print(f"  FS2 Layer 13 K shape: {fs2_layer13_k.shape}")
    print(f"  FS2 Layer 13 V shape: {fs2_layer13_v.shape}")
    print(f"  FS2 Layer 14 K shape: {fs2_layer14_k.shape}")
    print(f"  FS2 Layer 14 V shape: {fs2_layer14_v.shape}")

# ===== COMPARISON =====
print("\n" + "=" * 80)
print("K/V TENSOR COMPARISON")
print("=" * 80)

print("\nLayer 13 (LOCAL SOURCE):")
k13_diff = (hf_layer13_k - fs2_layer13_k).abs()
v13_diff = (hf_layer13_v - fs2_layer13_v).abs()
print(f"  K tensor diff - Max: {k13_diff.max().item():.6e}, Mean: {k13_diff.mean().item():.6e}")
print(f"  V tensor diff - Max: {v13_diff.max().item():.6e}, Mean: {v13_diff.mean().item():.6e}")

print("\nLayer 14 (GLOBAL SOURCE):")
k14_diff = (hf_layer14_k - fs2_layer14_k).abs()
v14_diff = (hf_layer14_v - fs2_layer14_v).abs()
print(f"  K tensor diff - Max: {k14_diff.max().item():.6e}, Mean: {k14_diff.mean().item():.6e}")
print(f"  V tensor diff - Max: {v14_diff.max().item():.6e}, Mean: {v14_diff.mean().item():.6e}")

# Determine status
print("\n" + "=" * 80)
THRESHOLD = 1e-4

all_good = True
if k13_diff.max().item() > THRESHOLD or v13_diff.max().item() > THRESHOLD:
    print("❌ Layer 13 K/V tensors DIVERGE")
    all_good = False
else:
    print("✅ Layer 13 K/V tensors MATCH")

if k14_diff.max().item() > THRESHOLD or v14_diff.max().item() > THRESHOLD:
    print("❌ Layer 14 K/V tensors DIVERGE")
    all_good = False
else:
    print("✅ Layer 14 K/V tensors MATCH")

if all_good:
    print("\n⚠️  SOURCE layers store identical K/V, but parity still breaks at layer 15!")
    print("    Issue must be in how CONSUMER layers USE the stored K/V.")
else:
    print("\n⚠️  SOURCE layers store DIFFERENT K/V!")
    print("    Need to debug why stored K/V differs.")

print("=" * 80)

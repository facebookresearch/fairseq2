#!/usr/bin/env python3
"""Compare K/V stored by layer 18 in HF vs FS2 directly."""

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
print("RUNNING HF THROUGH LAYER 18")
print("=" * 80)

with torch.no_grad():
    hf_lm = hf_model.model.language_model

    # Get embeddings
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
    hf_hidden = torch.stack(temp_hidden_states, dim=0)

    # Position embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(hf_lm.config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden, position_ids, layer_type)

    # Get PLE
    state_bag = IncrementalStateBag(input_ids.shape[1])
    batch_layout = BatchLayout(input_ids.shape, [input_ids.shape[1]], device=device)
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    per_layer_inputs = state_bag.per_layer_inputs

    # HF cache
    hf_cache = DynamicCache()

    # Run HF layers 0-18
    print("Running HF layers 0-18...")
    for idx in range(19):
        layer = hf_lm.layers[idx]
        layer_type = layer.attention_type
        layer_ple = per_layer_inputs[:, :, idx, :]

        hf_hidden = layer(
            hf_hidden,
            position_embeddings=position_embeddings[layer_type],
            per_layer_input=layer_ple,
            past_key_values=hf_cache,  # Fixed: plural
        )

        if idx == 18:
            print(f"\nAfter HF layer 18:")
            print(f"  Hidden state shape: {hf_hidden.shape}")
            print(f"  Hidden[0] stats: min={hf_hidden[0].min().item():.4f}, max={hf_hidden[0].max().item():.4f}")

    # Check if HF stored K/V
    print(f"\nChecking HF cache for stored K/V...")
    print(f"  Has 'shared_layers': {hasattr(hf_cache, 'shared_layers')}")

    if hasattr(hf_cache, 'shared_layers'):
        print(f"  shared_layers keys: {list(hf_cache.shared_layers.keys())}")
        if 18 in hf_cache.shared_layers:
            hf_k, hf_v = hf_cache.shared_layers[18]
            print(f"  ✅ HF layer 18 stored K/V")
            print(f"     K shape: {hf_k.shape}")
            print(f"     V shape: {hf_v.shape}")
            print(f"     K stats: min={hf_k.min().item():.4f}, max={hf_k.max().item():.4f}")
            print(f"     V stats: min={hf_v.min().item():.4f}, max={hf_v.max().item():.4f}")
        else:
            print(f"  ❌ Layer 18 NOT in shared_layers!")
            hf_k, hf_v = None, None
    else:
        print(f"  ❌ No shared_layers attribute!")
        hf_k, hf_v = None, None

print("\n" + "=" * 80)
print("RUNNING FS2 THROUGH LAYER 18")
print("=" * 80)

with torch.no_grad():
    fs2_hidden = fs2_model.decoder._stack_altup(fs2_embeds)

    kv_slots = {
        KVProjectionType.LOCAL: None,
        KVProjectionType.GLOBAL: None,
    }

    attn_bias_cache = AttentionBiasCache()

    # Run FS2 layers 0-18
    print("Running FS2 layers 0-18...")
    for idx in range(19):
        layer = fs2_model.decoder.layers[idx]
        layer_ple = per_layer_inputs[:, :, idx, :]

        fs2_hidden = layer(
            fs2_hidden, batch_layout, attn_bias_cache,
            per_layer_input=layer_ple,
            state_bag=None,
            kv_projection_slots=kv_slots
        )

        if idx == 18:
            print(f"\nAfter FS2 layer 18:")
            print(f"  Hidden state shape: {fs2_hidden.shape}")
            print(f"  Hidden[0] stats: min={fs2_hidden[0].min().item():.4f}, max={fs2_hidden[0].max().item():.4f}")

    # Check if FS2 stored K/V
    print(f"\nChecking FS2 slots for stored K/V...")
    if kv_slots[KVProjectionType.LOCAL] is not None:
        fs2_k, fs2_v = kv_slots[KVProjectionType.LOCAL]
        print(f"  ✅ FS2 layer 18 stored K/V in LOCAL slot")
        print(f"     K shape: {fs2_k.shape}")
        print(f"     V shape: {fs2_v.shape}")
        print(f"     K stats: min={fs2_k.min().item():.4f}, max={fs2_k.max().item():.4f}")
        print(f"     V stats: min={fs2_v.min().item():.4f}, max={fs2_v.max().item():.4f}")
    else:
        print(f"  ❌ LOCAL slot is empty!")
        fs2_k, fs2_v = None, None

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if hf_k is None or fs2_k is None:
    print("❌ Cannot compare - one or both didn't store K/V!")
else:
    print(f"\nShape comparison:")
    print(f"  HF K:  {hf_k.shape}  (format: batch, num_kv_heads, seq_len, head_dim)")
    print(f"  FS2 K: {fs2_k.shape}  (format: batch, seq_len, num_kv_heads, head_dim)")

    # Transpose FS2 to match HF format
    fs2_k_transposed = fs2_k.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)
    fs2_v_transposed = fs2_v.transpose(1, 2)

    print(f"\nAfter transposing FS2:")
    print(f"  FS2 K transposed: {fs2_k_transposed.shape}")

    if hf_k.shape != fs2_k_transposed.shape:
        print(f"\n❌ SHAPE MISMATCH even after transpose!")
    else:
        k_diff = (hf_k - fs2_k_transposed).abs()
        v_diff = (hf_v - fs2_v_transposed).abs()

        print(f"\nK difference:")
        print(f"  Max:  {k_diff.max().item():.6e}")
        print(f"  Mean: {k_diff.mean().item():.6e}")
        print(f"  Median: {k_diff.median().item():.6e}")

        print(f"\nV difference:")
        print(f"  Max:  {v_diff.max().item():.6e}")
        print(f"  Mean: {v_diff.mean().item():.6e}")
        print(f"  Median: {v_diff.median().item():.6e}")

        print("\n" + "=" * 80)
        if k_diff.max().item() < 1e-4 and v_diff.max().item() < 1e-4:
            print("✅ STORED K/V MATCH!")
            print("   Layer 18 stores identical K/V in both implementations.")
        else:
            print("❌ STORED K/V DIVERGE!")
            print("   This explains why layer 20 (consumer) produces wrong output.")
            print("\n   Possible causes:")
            print("   1. Different input to layer 18 (check hidden state diff)")
            print("   2. Different K/V projection weights")
            print("   3. Different normalization (k_norm, v_norm)")
            print("   4. Different RoPE application")
            print("   5. Storing at wrong point in computation")

print("=" * 80)

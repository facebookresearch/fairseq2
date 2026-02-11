#!/usr/bin/env python3
"""Compare actual K/V values stored by layer 18 in HF vs FS2."""

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

print(f"Input: {text!r}\n")

with torch.no_grad():
    # ===== HF =====
    print("[1/2] Running HF through layer 18...")
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

    # Run HF layers 0-18 with cache to trigger storage
    hf_cache = DynamicCache()
    for idx in range(19):
        layer = hf_lm.layers[idx]
        layer_type = layer.attention_type
        layer_ple = per_layer_inputs[:, :, idx, :]

        hf_hidden = layer(
            hf_hidden,
            position_embeddings=position_embeddings[layer_type],
            per_layer_input=layer_ple,
            past_key_value=hf_cache,
        )

    # Extract stored K/V from HF
    print("  Extracting HF stored K/V from layer 18...")

    if not hasattr(hf_cache, 'shared_layers'):
        print("\n❌ HF cache does NOT have 'shared_layers' attribute!")
        print("   This means layer 18 did NOT store K/V.")
        print(f"   Cache attributes: {[a for a in dir(hf_cache) if not a.startswith('_')]}")
        raise RuntimeError("Layer 18 should have stored to shared_layers but didn't")

    if 18 not in hf_cache.shared_layers:
        print(f"\n❌ Layer 18 not in shared_layers!")
        print(f"   Available layers: {list(hf_cache.shared_layers.keys())}")
        raise RuntimeError("Layer 18 should be in shared_layers")

    hf_k, hf_v = hf_cache.shared_layers[18]
    print(f"  HF K shape: {hf_k.shape}")
    print(f"  HF V shape: {hf_v.shape}")

    # ===== FS2 =====
    print("\n[2/2] Running FS2 through layer 18...")
    fs2_hidden = fs2_model.decoder._stack_altup(fs2_embeds)

    kv_slots = {
        KVProjectionType.LOCAL: None,
        KVProjectionType.GLOBAL: None,
    }

    attn_bias_cache = AttentionBiasCache()
    for idx in range(19):
        layer = fs2_model.decoder.layers[idx]
        layer_ple = per_layer_inputs[:, :, idx, :]

        fs2_hidden = layer(
            fs2_hidden, batch_layout, attn_bias_cache,
            per_layer_input=layer_ple,
            state_bag=None,
            kv_projection_slots=kv_slots
        )

    # Extract stored K/V from FS2
    print("  Extracting FS2 stored K/V from LOCAL slot...")
    fs2_k, fs2_v = kv_slots[KVProjectionType.LOCAL]
    print(f"  FS2 K shape: {fs2_k.shape}")
    print(f"  FS2 V shape: {fs2_v.shape}")

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARING STORED K/V FROM LAYER 18")
print("="*80)

print(f"\nHF stores in format: (batch, num_kv_heads, seq_len, head_dim)")
print(f"FS2 stores in format: (batch, seq_len, num_kv_heads, head_dim)")
print(f"\nNeed to transpose FS2 to match HF format for comparison...")

# Transpose FS2 from (B, S, H, D) to (B, H, S, D)
fs2_k_transposed = fs2_k.transpose(1, 2)  # (1, 5, 2, 256) -> (1, 2, 5, 256)
fs2_v_transposed = fs2_v.transpose(1, 2)

print(f"\nFS2 K transposed shape: {fs2_k_transposed.shape}")
print(f"HF K shape: {hf_k.shape}")

if fs2_k_transposed.shape != hf_k.shape:
    print("\n❌ SHAPE MISMATCH!")
else:
    k_diff = (hf_k - fs2_k_transposed).abs()
    v_diff = (hf_v - fs2_v_transposed).abs()

    print(f"\nK difference:")
    print(f"  Max:  {k_diff.max().item():.6e}")
    print(f"  Mean: {k_diff.mean().item():.6e}")

    print(f"\nV difference:")
    print(f"  Max:  {v_diff.max().item():.6e}")
    print(f"  Mean: {v_diff.mean().item():.6e}")

    if k_diff.max().item() < 1e-4 and v_diff.max().item() < 1e-4:
        print("\n✅ Stored K/V MATCH between HF and FS2!")
    else:
        print("\n❌ Stored K/V DIVERGE between HF and FS2!")
        print("   This explains why consumer layers produce wrong results.")

print("="*80)

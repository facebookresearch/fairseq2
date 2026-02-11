#!/usr/bin/env python3
"""Test if attention needs isolated state_bag."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

    # Setup with frontend
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    # Get to layer 0 input (manually)
    fs2_layer = fs2_model.decoder.layers[0]
    predictions = fs2_layer.altup(fs2_hidden_4d)
    active = predictions[0]
    normed = fs2_layer.input_layernorm(active)

    print("="*80)
    print("Attention with SAME state_bag as frontend")
    print("="*80)

    attn_bias_cache = AttentionBiasCache()
    attn_same = fs2_layer.self_attn(
        normed, batch_layout, normed, batch_layout, normed,
        attn_bias_cache, state_bag=state_bag  # Same state_bag
    )
    print(f"Attention output: mean={attn_same.mean():.6f}, std={attn_same.std():.6f}")

    print("\n" + "="*80)
    print("Attention with FRESH state_bag")
    print("="*80)

    fresh_state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache2 = AttentionBiasCache()
    attn_fresh = fs2_layer.self_attn(
        normed, batch_layout, normed, batch_layout, normed,
        attn_bias_cache2, state_bag=fresh_state_bag  # Fresh state_bag
    )
    print(f"Attention output: mean={attn_fresh.mean():.6f}, std={attn_fresh.std():.6f}")

    print("\n" + "="*80)
    print("Difference")
    print("="*80)
    diff = (attn_same - attn_fresh).abs()
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")

    if diff.max() > 1e-6:
        print("\n⚠️  Attention is affected by state_bag contents!")
        print("State_bag from frontend contains:")
        for key in vars(state_bag):
            if not key.startswith('_'):
                val = getattr(state_bag, key)
                print(f"  {key}: {type(val)}, shape={getattr(val, 'shape', 'N/A')}")
    else:
        print("\n✓ Attention not affected by state_bag")

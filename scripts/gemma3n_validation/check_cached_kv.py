#!/usr/bin/env python3
"""Check if FS2 is using cached K/V from state_bag instead of fresh computation."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("CHECK IF FS2 IS USING CACHED K/V")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    fs2_layer0 = fs2_model.decoder.layers[0]

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    # Hook to check if cached K/V is used
    kv_source_log = []

    original_get_kv = fs2_layer0.self_attn._project_kv

    def project_kv_hook(keys, keys_layout, values, state_bag=None):
        kv_source_log.append("FRESH K/V computed")
        return original_get_kv(keys, keys_layout, values, state_bag)

    fs2_layer0.self_attn._project_kv = project_kv_hook

    # Test 1: With fresh state_bag
    print("\n[TEST 1: Fresh state_bag]")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    # Check state_bag before
    from fairseq2.models.transformer.multihead_attention import AttentionState
    state_before = state_bag.maybe_get_state(fs2_layer0.self_attn, AttentionState)
    print(f"  State before: {state_before}")

    kv_source_log.clear()
    fs2_attn_out1 = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    print(f"  K/V source: {kv_source_log}")
    print(f"  Output[0, 3, :5]: {fs2_attn_out1[0, 3, :5]}")

    # Check state_bag after
    state_after = state_bag.maybe_get_state(fs2_layer0.self_attn, AttentionState)
    print(f"  State after: {'Cached' if state_after is not None else 'None'}")

    # Test 2: Reuse same state_bag (should use cached K/V!)
    print("\n[TEST 2: Reuse state_bag with cached K/V]")
    kv_source_log.clear()
    fs2_attn_out2 = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,  # SAME state_bag!
    )

    print(f"  K/V source: {kv_source_log if kv_source_log else 'CACHED K/V used (hook not called)!'}")
    print(f"  Output[0, 3, :5]: {fs2_attn_out2[0, 3, :5]}")

    # Compare
    diff = (fs2_attn_out1 - fs2_attn_out2).abs()
    print(f"\n  Diff between fresh and cached: max={diff.max().item():.6e}")

    fs2_layer0.self_attn._project_kv = original_get_kv

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if len(kv_source_log) == 0:
    print("⚠️  Second call used CACHED K/V (hook not called)")
    print("   If we accidentally reuse state_bag across tests, we'd get stale K/V!")
else:
    print("✓ Both calls computed fresh K/V")

print("="*80)

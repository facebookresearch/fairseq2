#!/usr/bin/env python3
"""
Test: Verify that removing scaling fixes the divergence.
We'll manually cancel out SDPA's 1/sqrt(d_k) scaling by pre-scaling Q.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
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
print("TEST: Disable SDPA scaling by pre-scaling Q")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup (same as direct_attention_comparison.py)
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

    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    # Run HF attention (normal)
    print(f"\n[HF ATTENTION - scaling=1.0]")
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    print(f"Output[0, 3, :5]: {hf_attn_out[0, 3, :5]}")

    # Run FS2 attention WITH scaling fix
    print(f"\n[FS2 ATTENTION - with scaling fix]")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    # HACK: Patch _project_q to pre-scale Q by sqrt(d_k) to cancel SDPA's 1/sqrt(d_k)
    original_project_q = fs2_layer0.self_attn._project_q
    head_dim = 256

    def patched_project_q(seqs, seqs_layout, state_bag=None):
        q = original_project_q(seqs, seqs_layout, state_bag)
        # Pre-multiply by sqrt(d_k) to cancel SDPA's division
        q = q * (head_dim ** 0.5)
        return q

    fs2_layer0.self_attn._project_q = patched_project_q

    fs2_attn_out_fixed = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )
    print(f"Output[0, 3, :5]: {fs2_attn_out_fixed[0, 3, :5]}")

    # Restore
    fs2_layer0.self_attn._project_q = original_project_q

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    diff = (hf_attn_out - fs2_attn_out_fixed).abs()
    print(f"\nMax diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")

    if diff.max().item() < 1e-5:
        print(f"\n✅ OUTPUTS MATCH! Scaling was the root cause!")
    else:
        print(f"\n❌ Still diverges - scaling not the only issue")

print("="*80)

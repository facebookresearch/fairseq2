#!/usr/bin/env python3
"""Directly test NaiveSDPA with captured Q, K, V from real forward pass."""

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
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("TEST NaiveSDPA DIRECTLY WITH REAL Q, K, V")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Full setup to get real Q, K, V
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

    # Capture Q, K, V from FS2 forward pass
    captured = {}

    original_sdpa = fs2_layer0.self_attn.sdpa.forward
    def capture_hook(q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
        captured['q'] = q.clone()
        captured['k'] = k.clone()
        captured['v'] = v.clone()
        captured['q_layout'] = q_layout
        captured['k_layout'] = k_layout
        captured['bias_cache'] = bias_cache
        return original_sdpa(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)

    fs2_layer0.self_attn.sdpa.forward = capture_hook

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    fs2_attn_out_real = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn.sdpa.forward = original_sdpa

    print(f"\nCaptured Q, K, V from real forward:")
    print(f"  Q: {captured['q'].shape}")
    print(f"  K: {captured['k'].shape}")
    print(f"  V: {captured['v'].shape}")

    # Now call NaiveSDPA directly with these captured inputs
    print(f"\n[CALLING NaiveSDPA DIRECTLY]")
    from fairseq2.models.transformer.sdpa.naive import naive_scaled_dot_product_attention
    from fairseq2.models.transformer.attention_bias import maybe_get_attention_bias_tensor

    q = captured['q']
    k = captured['k']
    v = captured['v']

    # Get bias
    bias = maybe_get_attention_bias_tensor(
        fs2_layer0.self_attn.sdpa.bias, q, captured['q_layout'], captured['k_layout'], captured['bias_cache']
    )

    print(f"  Bias shape: {bias.shape if bias is not None else None}")

    # Transpose for naive_scaled_dot_product_attention (expects [batch, heads, seq, dim])
    q_t = q.transpose(-2, -3)
    k_t = k.transpose(-2, -3)
    v_t = v.transpose(-2, -3)

    print(f"  Calling naive_scaled_dot_product_attention")
    print(f"    Q: {q_t.shape}, K: {k_t.shape}, V: {v_t.shape}")

    attns_direct, _ = naive_scaled_dot_product_attention(q_t, k_t, v_t, bias, dropout_p=0.0)

    # Transpose back
    attns_direct = attns_direct.transpose(-2, -3)

    print(f"  Direct SDPA output: {attns_direct.shape}")
    print(f"  Direct SDPA[0, 3, 3, :5]: {attns_direct[0, 3, 3, :5]}")

    # Apply output projection
    batch, seq, heads, dim = attns_direct.shape
    attns_flat = attns_direct.reshape(batch, seq, heads * dim)
    fs2_attn_out_manual = fs2_layer0.self_attn.output_proj(attns_flat)

    print(f"  After output_proj: {fs2_attn_out_manual.shape}")
    print(f"  Manual output[0, 3, :5]: {fs2_attn_out_manual[0, 3, :5]}")

    #Compare with real FS2 output
    print(f"\n[COMPARISON]")
    print(f"  Real FS2 output[0, 3, :5]: {fs2_attn_out_real[0, 3, :5]}")

    diff = (fs2_attn_out_manual - fs2_attn_out_real).abs()
    print(f"  Diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    if diff.max().item() < 1e-6:
        print(f"\n✅ Manual call matches real forward pass")
        print(f"   NaiveSDPA itself is working correctly")
    else:
        print(f"\n❌ Manual call differs from real forward pass")
        print(f"   Something is different in the real execution")

print("="*80)

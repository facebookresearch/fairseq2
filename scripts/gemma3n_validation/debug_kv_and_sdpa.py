#!/usr/bin/env python3
"""Debug K, V, and SDPA to find divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("K, V, AND SDPA DEBUG")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup (reuse from previous script)
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

    hf_attn = hf_layer0.self_attn
    fs2_attn = fs2_layer0.self_attn

    bsz, q_len, _ = hf_active_normed.shape
    num_heads = hf_attn.config.num_attention_heads
    num_kv_heads = hf_attn.config.num_key_value_heads
    head_dim = hf_attn.config.head_dim

    # K projection
    print(f"\n[K PROJECTION]")
    hf_k = hf_attn.k_proj(hf_active_normed)
    fs2_k = fs2_attn.k_proj(fs2_active_normed)

    hf_k_reshaped = hf_k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    fs2_k_reshaped = fs2_k.view(bsz, q_len, num_kv_heads, head_dim)

    diff = (hf_k_reshaped.transpose(1, 2) - fs2_k_reshaped).abs()
    print(f"  K projection diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # K normalization
    hf_k_normed = hf_attn.k_norm(hf_k_reshaped)
    fs2_k_normed = fs2_attn.k_norm(fs2_k_reshaped)

    diff = (hf_k_normed.transpose(1, 2) - fs2_k_normed).abs()
    print(f"  K normalized diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # K with RoPE
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    hf_cos, hf_sin = position_embeddings[hf_layer0.attention_type]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    hf_k_with_rope = (hf_k_normed * hf_cos) + (rotate_half(hf_k_normed) * hf_sin)
    fs2_k_with_rope = fs2_attn.pos_encoder(fs2_k_normed, batch_layout, state_bag=state_bag)

    diff = (hf_k_with_rope.transpose(1, 2) - fs2_k_with_rope).abs()
    print(f"  K with RoPE diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # V projection
    print(f"\n[V PROJECTION]")
    hf_v = hf_attn.v_proj(hf_active_normed)
    fs2_v = fs2_attn.v_proj(fs2_active_normed)

    hf_v_reshaped = hf_v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    fs2_v_reshaped = fs2_v.view(bsz, q_len, num_kv_heads, head_dim)

    diff = (hf_v_reshaped.transpose(1, 2) - fs2_v_reshaped).abs()
    print(f"  V projection diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # V normalization
    hf_v_normed = hf_attn.v_norm(hf_v_reshaped)
    fs2_v_normed = fs2_attn.v_norm(fs2_v_reshaped)

    diff = (hf_v_normed.transpose(1, 2) - fs2_v_normed).abs()
    print(f"  V normalized diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)

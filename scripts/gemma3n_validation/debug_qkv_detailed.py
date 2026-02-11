#!/usr/bin/env python3
"""Debug Q/K/V inside attention to find where they diverge."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")

# Load models
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

# Prepare input
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

# Get to attention input (LAuReL output) - reuse logic from debug_layer_steps
with torch.no_grad():
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # HF 4D stacking
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    hf_layer = hf_lm.layers[0]
    hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[0]
    hf_active_normed = hf_layer.input_layernorm(hf_active)
    hf_laurel = hf_layer.laurel(hf_active_normed)

    # FS2 setup
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(
        fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
    )

    fs2_layer = fs2_model.decoder.layers[0]
    fs2_predictions = fs2_layer.altup(fs2_hidden_4d)
    fs2_active = fs2_predictions[0]
    fs2_active_normed = fs2_layer.input_layernorm(fs2_active)
    fs2_laurel = fs2_layer.laurel(fs2_active_normed)

    # Now manually compute Q/K/V for both
    print("="*80)
    print("Manual Q/K/V Computation")
    print("="*80)

    # HF Q/K/V
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding, apply_rotary_pos_emb
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_laurel, position_ids, "sliding_attention")

    input_shape = hf_laurel.shape[:-1]
    hidden_shape = (*input_shape, -1, config.head_dim)

    hf_q = hf_layer.self_attn.q_proj(hf_laurel).view(hidden_shape)
    hf_q = hf_layer.self_attn.q_norm(hf_q)
    hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2)
    print(f"HF Q: mean={hf_q.mean():.6f}, std={hf_q.std():.6f}")

    hf_k = hf_layer.self_attn.k_proj(hf_laurel).view(hidden_shape)
    hf_k = hf_layer.self_attn.k_norm(hf_k)
    hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2)
    print(f"HF K: mean={hf_k.mean():.6f}, std={hf_k.std():.6f}")

    hf_v = hf_layer.self_attn.v_proj(hf_laurel).view(hidden_shape)
    hf_v = hf_layer.self_attn.v_norm(hf_v)
    print(f"HF V: mean={hf_v.mean():.6f}, std={hf_v.std():.6f}")

    # FS2 Q/K/V
    attn_state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_q = fs2_layer.self_attn._project_q(fs2_laurel, batch_layout, attn_state_bag)
    print(f"\nFS2 Q: mean={fs2_q.mean():.6f}, std={fs2_q.std():.6f}")

    fs2_k, fs2_v = fs2_layer.self_attn._project_kv(fs2_laurel, batch_layout, fs2_laurel, attn_state_bag)
    print(f"FS2 K: mean={fs2_k.mean():.6f}, std={fs2_k.std():.6f}")
    print(f"FS2 V: mean={fs2_v.mean():.6f}, std={fs2_v.std():.6f}")

    # Compare
    print("\n" + "="*80)
    print("Q/K/V Comparison")
    print("="*80)
    q_diff = (hf_q - fs2_q).abs()
    k_diff = (hf_k - fs2_k).abs()
    v_diff = (hf_v - fs2_v).abs()
    print(f"Q diff: max={q_diff.max():.6e}, mean={q_diff.mean():.6e}")
    print(f"K diff: max={k_diff.max():.6e}, mean={k_diff.mean():.6e}")
    print(f"V diff: max={v_diff.max():.6e}, mean={v_diff.mean():.6e}")

    # Now call attention via the layer's forward vs manually
    print("\n" + "="*80)
    print("Attention via layer forward")
    print("="*80)

    hf_attn, _ = hf_layer.self_attn(
        hidden_states=hf_laurel,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )
    print(f"HF attention: mean={hf_attn.mean():.6f}, std={hf_attn.std():.6f}")

    attn_bias_cache = AttentionBiasCache()
    attn_state_bag2 = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_attn = fs2_layer.self_attn(
        fs2_laurel, batch_layout, fs2_laurel, batch_layout, fs2_laurel,
        attn_bias_cache, state_bag=attn_state_bag2
    )
    print(f"FS2 attention: mean={fs2_attn.mean():.6f}, std={fs2_attn.std():.6f}")

    attn_diff = (hf_attn - fs2_attn).abs()
    print(f"\nAttention diff: max={attn_diff.max():.6e}, mean={attn_diff.mean():.6e}")

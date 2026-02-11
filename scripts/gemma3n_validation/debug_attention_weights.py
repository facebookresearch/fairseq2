#!/usr/bin/env python3
"""Compare attention weights to find SDPA divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

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
    # Setup (same as before)
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
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

    # Manual attention computation
    from transformers.models.gemma3n.modeling_gemma3n import (
        Gemma3nRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
    )
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_laurel, position_ids, "sliding_attention")

    input_shape = hf_laurel.shape[:-1]
    hidden_shape = (*input_shape, -1, config.head_dim)

    # HF manual SDPA
    hf_q = hf_layer.self_attn.q_proj(hf_laurel).view(hidden_shape)
    hf_q = hf_layer.self_attn.q_norm(hf_q)
    hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    hf_k = hf_layer.self_attn.k_proj(hf_laurel).view(hidden_shape)
    hf_k = hf_layer.self_attn.k_norm(hf_k)
    hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    hf_v = hf_layer.self_attn.v_proj(hf_laurel).view(hidden_shape)
    hf_v = hf_layer.self_attn.v_norm(hf_v).transpose(1, 2)

    hf_k = repeat_kv(hf_k, 4)
    hf_v = repeat_kv(hf_v, 4)

    scaling = config.head_dim ** -0.5
    hf_attn_weights = torch.matmul(hf_q, hf_k.transpose(2, 3)) * scaling
    print("HF attention weights (pre-softmax):")
    print(f"  mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {hf_attn_weights[0,0,0,:]}")

    # Apply mask
    seq_len = hf_q.size(2)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    hf_attn_weights = hf_attn_weights + causal_mask
    print(f"\nHF attention weights (post-mask, pre-softmax):")
    print(f"  mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {hf_attn_weights[0,0,0,:]}")

    hf_attn_weights = torch.nn.functional.softmax(hf_attn_weights, dim=-1, dtype=torch.float32).to(hf_q.dtype)
    print(f"\nHF attention probs (post-softmax):")
    print(f"  mean={hf_attn_weights.mean():.6f}, std={hf_attn_weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {hf_attn_weights[0,0,0,:]}")

    hf_attn_manual = torch.matmul(hf_attn_weights, hf_v).transpose(1, 2).contiguous()
    hf_attn_manual = hf_attn_manual.reshape(*input_shape, -1)
    hf_attn_manual = hf_layer.self_attn.o_proj(hf_attn_manual)

    print(f"\nHF manual attention output: mean={hf_attn_manual.mean():.6f}, std={hf_attn_manual.std():.6f}")

    # FS2 via NaiveSDPA - need to intercept attention weights
    # Use the SDPA directly
    from fairseq2.models.transformer.sdpa.naive import naive_scaled_dot_product_attention
    from fairseq2.nn.utils.mask import repeat_interleave

    attn_state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_q = fs2_layer.self_attn._project_q(fs2_laurel, batch_layout, attn_state_bag)
    fs2_k, fs2_v = fs2_layer.self_attn._project_kv(fs2_laurel, batch_layout, fs2_laurel, attn_state_bag)

    # Repeat for GQA
    fs2_k = repeat_interleave(fs2_k, dim=-2, repeat=4)
    fs2_v = repeat_interleave(fs2_v, dim=-2, repeat=4)

    # Transpose for SDPA
    fs2_q = fs2_q.transpose(-2, -3)
    fs2_k = fs2_k.transpose(-2, -3)
    fs2_v = fs2_v.transpose(-2, -3)

    # Get bias
    from fairseq2.models.transformer import CausalAttentionBias
    bias = CausalAttentionBias(attn_window_len=512)
    bias_tensor = bias.create_bias_tensor(seq_len, seq_len, device, torch.float32)

    print("\n" + "="*80)
    print("FS2 SDPA computation")
    print("="*80)

    # Manually compute to see weights
    weights = torch.matmul(fs2_q, fs2_k.transpose(-1, -2))
    weights = weights * (config.head_dim ** -0.5)
    print(f"FS2 attention weights (pre-bias): mean={weights.mean():.6f}, std={weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {weights[0,0,0,:]}")

    weights = weights + bias_tensor
    print(f"\nFS2 attention weights (post-bias, pre-softmax): mean={weights.mean():.6f}, std={weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {weights[0,0,0,:]}")

    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(fs2_q.dtype)
    print(f"\nFS2 attention probs (post-softmax): mean={weights.mean():.6f}, std={weights.std():.6f}")
    print(f"  sample[0,0,0,:]: {weights[0,0,0,:]}")

    fs2_attns = torch.matmul(weights, fs2_v)
    fs2_attns = fs2_attns.transpose(-2, -3)
    fs2_attns = fs2_attns.flatten(-2, -1)
    fs2_attns = fs2_layer.self_attn.output_proj(fs2_attns)

    print(f"\nFS2 manual attention output: mean={fs2_attns.mean():.6f}, std={fs2_attns.std():.6f}")

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    diff = (hf_attn_manual - fs2_attns).abs()
    print(f"Attention output diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

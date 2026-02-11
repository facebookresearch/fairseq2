#!/usr/bin/env python3
"""Compare Gaussian top-k implementation: HF vs FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("Gaussian Top-K Comparison: HF vs FS2")
print("="*80)

with torch.no_grad():
    # Setup
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # 4D stack
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=dtype)
    temp_hidden = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs_discrete)

    # Get layers
    hf_layer = hf_lm.layers[0]
    fs2_layer = fs2_model.decoder.layers[0]

    # Trace to FFN input
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    # Get to FFN input (abbreviated)
    hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[0]
    hf_normed = hf_layer.input_layernorm(hf_active)
    hf_laurel = hf_layer.laurel(hf_normed)

    # Attention
    input_shape = hf_normed.shape[:-1]
    hidden_shape = (*input_shape, -1, config.head_dim)
    hf_q = hf_layer.self_attn.q_proj(hf_normed).view(hidden_shape)
    hf_q = hf_layer.self_attn.q_norm(hf_q)
    hf_q = apply_rotary_pos_emb(hf_q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    hf_k = hf_layer.self_attn.k_proj(hf_normed).view(hidden_shape)
    hf_k = hf_layer.self_attn.k_norm(hf_k)
    hf_k = apply_rotary_pos_emb(hf_k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    hf_v = hf_layer.self_attn.v_proj(hf_normed).view(hidden_shape)
    hf_v = hf_layer.self_attn.v_norm(hf_v).transpose(1, 2)
    hf_k = repeat_kv(hf_k, 4)
    hf_v = repeat_kv(hf_v, 4)
    scaling = config.head_dim ** -0.5
    hf_attn_weights = torch.matmul(hf_q, hf_k.transpose(2, 3)) * scaling
    seq_len = hf_q.size(2)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    hf_attn_weights = hf_attn_weights + causal_mask
    hf_attn_weights = torch.nn.functional.softmax(hf_attn_weights, dim=-1, dtype=torch.float32).to(hf_q.dtype)
    hf_attn = torch.matmul(hf_attn_weights, hf_v).transpose(1, 2).contiguous()
    hf_attn = hf_attn.reshape(*input_shape, -1)
    hf_attn = hf_layer.self_attn.o_proj(hf_attn)

    hf_attn_normed = hf_layer.post_attention_layernorm(hf_attn)
    hf_attn_gated = hf_active + hf_attn_normed
    import math
    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)

    # FFN input
    hf_ffn_input = hf_layer.pre_feedforward_layernorm(hf_attn_laurel)

    print("\n[FFN INPUT]")
    print(f"Shape: {hf_ffn_input.shape}")
    print(f"Mean: {hf_ffn_input.mean():.6f}, Std: {hf_ffn_input.std():.6f}")

    # HF gate projection (before sparsification)
    hf_gate_before = hf_layer.mlp.gate_proj(hf_ffn_input)
    print(f"\n[HF GATE (before sparsification)]")
    print(f"Shape: {hf_gate_before.shape}")
    print(f"Mean: {hf_gate_before.mean():.6f}, Std: {hf_gate_before.std():.6f}")
    print(f"Min: {hf_gate_before.min():.6f}, Max: {hf_gate_before.max():.6f}")

    # FS2 gate projection (before sparsification)
    fs2_gate_before = fs2_layer.ffn.gate_proj(hf_ffn_input)  # Use same input
    print(f"\n[FS2 GATE (before sparsification)]")
    print(f"Shape: {fs2_gate_before.shape}")
    print(f"Mean: {fs2_gate_before.mean():.6f}, Std: {fs2_gate_before.std():.6f}")
    print(f"Min: {fs2_gate_before.min():.6f}, Max: {fs2_gate_before.max():.6f}")

    diff_before = (hf_gate_before - fs2_gate_before).abs()
    print(f"\n[GATE BEFORE SPARSIFICATION DIFF]")
    print(f"Max: {diff_before.max():.6e}, Mean: {diff_before.mean():.6e}")

    # Apply HF Gaussian top-k manually
    print(f"\n[HF GAUSSIAN TOP-K]")
    target_sparsity = torch.tensor(0.95, dtype=torch.float32, device=device)
    normal_dist = torch.distributions.normal.Normal(0, 1)
    std_multiplier = normal_dist.icdf(target_sparsity).to(hf_gate_before.dtype)
    print(f"Target sparsity: 0.95")
    print(f"Std multiplier: {std_multiplier:.6f}")

    gate_mean = torch.mean(hf_gate_before, dim=-1, keepdim=True)
    gate_std = torch.std(hf_gate_before, dim=-1, keepdim=True, unbiased=False)
    print(f"Gate mean (per-seq): {gate_mean.squeeze()}")
    print(f"Gate std (per-seq): {gate_std.squeeze()}")

    cutoff = gate_mean + gate_std * std_multiplier
    print(f"Cutoff (per-seq): {cutoff.squeeze()}")

    hf_gate_after = torch.nn.functional.relu(hf_gate_before - cutoff)
    print(f"\n[HF GATE (after sparsification)]")
    print(f"Mean: {hf_gate_after.mean():.6f}, Std: {hf_gate_after.std():.6f}")
    print(f"Nonzero ratio: {(hf_gate_after != 0).float().mean():.6f}")

    # Apply FS2 Gaussian top-k
    fs2_gate_after = fs2_layer.ffn._gaussian_topk(fs2_gate_before)
    print(f"\n[FS2 GATE (after sparsification)]")
    print(f"Mean: {fs2_gate_after.mean():.6f}, Std: {fs2_gate_after.std():.6f}")
    print(f"Nonzero ratio: {(fs2_gate_after != 0).float().mean():.6f}")

    diff_after = (hf_gate_after - fs2_gate_after).abs()
    print(f"\n[GATE AFTER SPARSIFICATION DIFF]")
    print(f"Max: {diff_after.max():.6e}, Mean: {diff_after.mean():.6e}")

    # Check if sparsification patterns match
    hf_mask = (hf_gate_after != 0)
    fs2_mask = (fs2_gate_after != 0)
    mask_match = (hf_mask == fs2_mask).float().mean()
    print(f"Sparsification mask match: {mask_match:.6f}")

    # Continue FFN forward
    print(f"\n[COMPLETE FFN]")
    hf_gate_activated = torch.nn.functional.gelu(hf_gate_after, approximate="tanh")
    hf_up = hf_layer.mlp.up_proj(hf_ffn_input)
    hf_ffn_output = hf_layer.mlp.down_proj(hf_gate_activated * hf_up)

    # Call FS2 FFN directly
    fs2_ffn_output = fs2_layer.ffn(hf_ffn_input)  # Use same input

    diff_ffn = (hf_ffn_output - fs2_ffn_output).abs()
    print(f"FFN output diff: max={diff_ffn.max():.6e}, mean={diff_ffn.mean():.6e}")

    print("\n" + "="*80)
    if diff_after.max() < 1e-6:
        print("✓ Gaussian top-k matches exactly")
    else:
        print("❌ Gaussian top-k differs - check implementation")
    print("="*80)

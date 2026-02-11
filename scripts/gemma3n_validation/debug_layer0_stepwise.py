#!/usr/bin/env python3
"""Step through layer 0 forward pass to find where divergence starts."""

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
    # Setup
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # PLE
    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs_discrete)
    hf_ple_layer0 = per_layer_inputs[:, :, 0, :]

    # 4D stacking
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    # FS2 setup
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(
        fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
    )
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)
    fs2_ple_layer0 = fs2_per_layer_inputs[:, :, 0, :]

    print("="*80)
    print("Step-by-step layer 0 forward pass")
    print("="*80)

    # HF layer 0
    hf_layer = hf_lm.layers[0]
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    # Step 1: AltUp predict
    hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_predictions = fs2_layer.altup(fs2_hidden_4d)
    diff = (hf_predictions - fs2_predictions).abs()
    print(f"\n1. AltUp predict: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 2: Extract active
    hf_active = hf_predictions[0]
    fs2_active = fs2_predictions[0]
    diff = (hf_active - fs2_active).abs()
    print(f"2. Active prediction: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 3: Input norm
    hf_normed = hf_layer.input_layernorm(hf_active)
    fs2_normed = fs2_layer.input_layernorm(fs2_active)
    diff = (hf_normed - fs2_normed).abs()
    print(f"3. Input layernorm: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 4: LAuReL
    hf_laurel = hf_layer.laurel(hf_normed)
    fs2_laurel = fs2_layer.laurel(fs2_normed)
    diff = (hf_laurel - fs2_laurel).abs()
    print(f"4. LAuReL: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 5: Attention (manual for HF)
    from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb, repeat_kv
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

    # FS2 attention
    attn_bias_cache = AttentionBiasCache()
    fs2_attn = fs2_layer.self_attn(
        fs2_normed, batch_layout, fs2_normed, batch_layout, fs2_normed,
        attn_bias_cache, state_bag=state_bag
    )

    diff = (hf_attn - fs2_attn).abs()
    print(f"5. Attention: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 6: Post-attention norm
    hf_attn_normed = hf_layer.post_attention_layernorm(hf_attn)
    fs2_attn_normed = fs2_layer.post_attention_layernorm(fs2_attn)
    diff = (hf_attn_normed - fs2_attn_normed).abs()
    print(f"6. Post-attention norm: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 7: Combine with LAuReL
    hf_attn_gated = hf_active + hf_attn_normed
    fs2_attn_gated = fs2_active + fs2_attn_normed
    diff = (hf_attn_gated - fs2_attn_gated).abs()
    print(f"7. Attn gated: max={diff.max():.6e}, mean={diff.mean():.6e}")

    import math
    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)
    fs2_attn_laurel = (fs2_attn_gated + fs2_laurel) / math.sqrt(2)
    diff = (hf_attn_laurel - fs2_attn_laurel).abs()
    print(f"8. Attn + LAuReL: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 8: FFN
    hf_attn_norm = hf_layer.pre_feedforward_layernorm(hf_attn_laurel)
    fs2_attn_norm = fs2_layer.pre_feedforward_layernorm(fs2_attn_laurel)
    diff = (hf_attn_norm - fs2_attn_norm).abs()
    print(f"9. Pre-FFN norm: max={diff.max():.6e}, mean={diff.mean():.6e}")

    hf_ffn = hf_layer.mlp(hf_attn_norm)
    fs2_ffn = fs2_layer.ffn(fs2_attn_norm)
    diff = (hf_ffn - fs2_ffn).abs()
    print(f"10. FFN: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 11: Post-FFN norm
    hf_ffn_norm = hf_layer.post_feedforward_layernorm(hf_ffn)
    fs2_ffn_norm = fs2_layer.post_feedforward_layernorm(fs2_ffn)
    diff = (hf_ffn_norm - fs2_ffn_norm).abs()
    print(f"11. Post-FFN norm: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 12: Final gating
    hf_attn_ffw_laurel_gated = hf_attn_laurel + hf_ffn_norm
    fs2_attn_ffw_laurel_gated = fs2_attn_laurel + fs2_ffn_norm
    diff = (hf_attn_ffw_laurel_gated - fs2_attn_ffw_laurel_gated).abs()
    print(f"12. Attn+FFN gated: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 13: AltUp correct
    hf_corrected = hf_layer.altup.correct(hf_predictions, hf_attn_ffw_laurel_gated)
    fs2_corrected = fs2_layer.altup.correct(fs2_predictions, fs2_attn_ffw_laurel_gated)
    diff = (hf_corrected - fs2_corrected).abs()
    print(f"13. AltUp correct: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 14: Extract first prediction and scale
    hf_first = hf_corrected[0].clone()
    fs2_first = fs2_corrected[0].clone()
    if hf_layer.config.altup_correct_scale:
        hf_first = hf_layer.altup.scale_corrected_output(hf_first)
    if fs2_layer.altup_correct_scale:
        fs2_first = fs2_layer.altup.scale_corrected_output(fs2_first)
    diff = (hf_first - fs2_first).abs()
    print(f"14. First prediction (scaled): max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 15: PLE gating
    hf_ple_gated = hf_layer.per_layer_input_gate(hf_first)
    fs2_ple_gated = fs2_layer.per_layer_input_gate(fs2_first)
    diff = (hf_ple_gated - fs2_ple_gated).abs()
    print(f"15. PLE gate proj: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 16: PLE activation
    hf_ple_act = hf_layer.act_fn(hf_ple_gated)
    import torch.nn as nn
    fs2_ple_act = fs2_layer.hidden_activation(fs2_ple_gated)
    diff = (hf_ple_act - fs2_ple_act).abs()
    print(f"16. PLE activation: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 17: PLE multiply
    hf_ple_mult = hf_ple_act * hf_ple_layer0
    fs2_ple_mult = fs2_ple_act * fs2_ple_layer0
    diff = (hf_ple_mult - fs2_ple_mult).abs()
    print(f"17. PLE multiply: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 18: PLE projection
    hf_ple_proj = hf_layer.per_layer_projection(hf_ple_mult)
    fs2_ple_proj = fs2_layer.per_layer_projection(fs2_ple_mult)
    diff = (hf_ple_proj - fs2_ple_proj).abs()
    print(f"18. PLE projection: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 19: PLE post-norm
    hf_ple_normed = hf_layer.post_per_layer_input_norm(hf_ple_proj)
    fs2_ple_normed = fs2_layer.post_per_layer_input_norm(fs2_ple_proj)
    diff = (hf_ple_normed - fs2_ple_normed).abs()
    print(f"19. PLE post-norm: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Step 20: Add to corrected[1:]
    hf_final = hf_corrected.clone()
    hf_final[1:] += hf_ple_normed
    fs2_final = fs2_corrected.clone()
    fs2_final[1:] += fs2_ple_normed
    diff = (hf_final - fs2_final).abs()
    print(f"20. Final output (4D): max={diff.max():.6e}, mean={diff.mean():.6e}")

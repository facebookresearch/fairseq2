#!/usr/bin/env python3
"""Side-by-side trace of layer 0 forward (HF vs FS2), showing only diffs."""

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

# Disable HF activation sparsity for deterministic comparison
for layer in hf_model.model.language_model.layers:
    if hasattr(layer, 'mlp'):
        layer.mlp.activation_sparsity = 0.0

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

def compare(name, hf_tensor, fs2_tensor, threshold=1e-5):
    diff = (hf_tensor - fs2_tensor).abs()
    max_diff = diff.max().item()
    if max_diff > threshold:
        print(f"❌ {name:25s} max={max_diff:.6e} | HF mean={hf_tensor.mean():.4f}, FS2 mean={fs2_tensor.mean():.4f}")
        return False
    else:
        print(f"✓  {name:25s} max={max_diff:.6e}")
        return True

print("="*80)
print("LAYER 0 SIDE-BY-SIDE TRACE (threshold=1e-5)")
print("="*80)

with torch.no_grad():
    # Setup HF
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
    hf_ple_layer0 = per_layer_inputs[:, :, 0, :]

    # Setup FS2
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)
    fs2_ple_layer0 = fs2_per_layer_inputs[:, :, 0, :]

    print("\n[INPUTS]")
    compare("4D hidden", hf_hidden_4d, fs2_hidden_4d, 1e-7)
    compare("PLE", hf_ple_layer0, fs2_ple_layer0, 1e-7)

    # Get layers
    hf_layer = hf_lm.layers[0]
    fs2_layer = fs2_model.decoder.layers[0]

    # Prepare RoPE for HF
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    print("\n[ALTUP PREDICT]")
    hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
    fs2_predictions = fs2_layer.altup(fs2_hidden_4d)
    compare("predictions (4D)", hf_predictions, fs2_predictions)

    hf_active = hf_predictions[0]
    fs2_active = fs2_predictions[0]
    compare("active (extracted)", hf_active, fs2_active)

    print("\n[PRE-ATTENTION]")
    hf_normed = hf_layer.input_layernorm(hf_active)
    fs2_normed = fs2_layer.input_layernorm(fs2_active)
    compare("input norm", hf_normed, fs2_normed)

    hf_laurel = hf_layer.laurel(hf_normed)
    fs2_laurel = fs2_layer.laurel(fs2_normed)
    compare("LAuReL", hf_laurel, fs2_laurel)

    print("\n[ATTENTION]")
    # HF attention (manual computation matching debug_layer0_stepwise.py)
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

    compare("attention output", hf_attn, fs2_attn)

    print("\n[POST-ATTENTION]")
    hf_attn_normed = hf_layer.post_attention_layernorm(hf_attn)
    fs2_attn_normed = fs2_layer.post_attention_layernorm(fs2_attn)
    compare("post-attn norm", hf_attn_normed, fs2_attn_normed)

    hf_attn_gated = hf_active + hf_attn_normed
    fs2_attn_gated = fs2_active + fs2_attn_normed
    compare("active + attn", hf_attn_gated, fs2_attn_gated)

    import math
    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)
    fs2_attn_laurel = (fs2_attn_gated + fs2_laurel) / math.sqrt(2)
    compare("LAuReL combine", hf_attn_laurel, fs2_attn_laurel)

    print("\n[FFN]")
    hf_attn_norm = hf_layer.pre_feedforward_layernorm(hf_attn_laurel)
    fs2_attn_norm = fs2_layer.pre_feedforward_layernorm(fs2_attn_laurel)
    compare("pre-FFN norm", hf_attn_norm, fs2_attn_norm)

    hf_ffn = hf_layer.mlp(hf_attn_norm)
    fs2_ffn = fs2_layer.ffn(fs2_attn_norm)
    compare("FFN output", hf_ffn, fs2_ffn)

    hf_ffn_norm = hf_layer.post_feedforward_layernorm(hf_ffn)
    fs2_ffn_norm = fs2_layer.post_feedforward_layernorm(fs2_ffn)
    compare("post-FFN norm", hf_ffn_norm, fs2_ffn_norm)

    hf_attn_ffw_laurel_gated = hf_attn_laurel + hf_ffn_norm
    fs2_attn_ffw_laurel_gated = fs2_attn_laurel + fs2_ffn_norm
    compare("LAuReL + FFN", hf_attn_ffw_laurel_gated, fs2_attn_ffw_laurel_gated)

    print("\n[ALTUP CORRECT]")
    hf_corrected = hf_layer.altup.correct(hf_predictions, hf_attn_ffw_laurel_gated)
    fs2_corrected = fs2_layer.altup.correct(fs2_predictions, fs2_attn_ffw_laurel_gated)
    compare("corrected (4D)", hf_corrected, fs2_corrected)

    hf_first = hf_corrected[0].clone()
    fs2_first = fs2_corrected[0].clone()
    if hf_layer.config.altup_correct_scale:
        hf_first = hf_layer.altup.scale_corrected_output(hf_first)
    if fs2_layer.altup_correct_scale:
        fs2_first = fs2_layer.altup.scale_corrected_output(fs2_first)
    compare("first (scaled)", hf_first, fs2_first)

    print("\n[PLE GATING]")
    hf_ple_gated = hf_layer.per_layer_input_gate(hf_first)
    fs2_ple_gated = fs2_layer.per_layer_input_gate(fs2_first)
    compare("PLE gate proj", hf_ple_gated, fs2_ple_gated)

    hf_ple_act = hf_layer.act_fn(hf_ple_gated)
    fs2_ple_act = fs2_layer.hidden_activation(fs2_ple_gated)
    compare("PLE activation", hf_ple_act, fs2_ple_act)

    hf_ple_mult = hf_ple_act * hf_ple_layer0
    fs2_ple_mult = fs2_ple_act * fs2_ple_layer0
    compare("PLE multiply", hf_ple_mult, fs2_ple_mult)

    hf_ple_proj = hf_layer.per_layer_projection(hf_ple_mult)
    fs2_ple_proj = fs2_layer.per_layer_projection(fs2_ple_mult)
    compare("PLE projection", hf_ple_proj, fs2_ple_proj)

    hf_ple_normed = hf_layer.post_per_layer_input_norm(hf_ple_proj)
    fs2_ple_normed = fs2_layer.post_per_layer_input_norm(fs2_ple_proj)
    compare("PLE post-norm", hf_ple_normed, fs2_ple_normed)

    # HF adds PLE to predictions[1:], NOT to [0]!
    hf_corrected[1:] += hf_ple_normed
    fs2_corrected[1:] += fs2_ple_normed

    print("\n[FINAL OUTPUT - ALL 4 PREDICTIONS]")
    for i in range(4):
        compare(f"prediction[{i}]", hf_corrected[i], fs2_corrected[i])

    print("\n" + "="*80)
    print("First divergence point shows the bug location")
    print("="*80)

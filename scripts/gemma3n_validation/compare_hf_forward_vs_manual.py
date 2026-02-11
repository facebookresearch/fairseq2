#!/usr/bin/env python3
"""Compare HF layer.forward() vs manual reconstruction."""

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
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("HF layer.forward() vs Manual Trace Comparison")
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

    # Get layer
    hf_layer = hf_lm.layers[0]

    # Prepare RoPE
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    print("\n[METHOD 1: Call hf_layer.forward() directly]")
    hf_output_forward = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=per_layer_inputs[:, :, 0, :],
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )
    print(f"Type: {type(hf_output_forward)}")
    print(f"Shape: {hf_output_forward.shape if isinstance(hf_output_forward, torch.Tensor) else 'N/A'}")

    # Check if it's a tuple
    if isinstance(hf_output_forward, tuple):
        print(f"Tuple length: {len(hf_output_forward)}")
        for i, elem in enumerate(hf_output_forward):
            if isinstance(elem, torch.Tensor):
                print(f"  Element {i}: shape={elem.shape}, mean={elem.mean():.6f}, std={elem.std():.6f}")
            else:
                print(f"  Element {i}: {type(elem)}")
        hf_output_actual = hf_output_forward[0] if isinstance(hf_output_forward[0], torch.Tensor) else hf_output_forward
    else:
        hf_output_actual = hf_output_forward
        print(f"Mean: {hf_output_actual.mean():.6f}")
        print(f"Std: {hf_output_actual.std():.6f}")

    print("\n[METHOD 2: Manual trace (from trace_layer0_sidebyside.py)]")

    # Manual computation (abbreviated from full trace)
    hf_predictions = hf_layer.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[0]
    hf_normed = hf_layer.input_layernorm(hf_active)
    hf_laurel = hf_layer.laurel(hf_normed)

    # Attention (manual)
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

    hf_attn_normed = hf_layer.post_attention_layernorm(hf_attn)
    hf_attn_gated = hf_active + hf_attn_normed

    import math
    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)

    # FFN
    hf_attn_norm = hf_layer.pre_feedforward_layernorm(hf_attn_laurel)
    hf_ffn = hf_layer.mlp(hf_attn_norm)
    hf_ffn_norm = hf_layer.post_feedforward_layernorm(hf_ffn)
    hf_attn_ffw_laurel_gated = hf_attn_laurel + hf_ffn_norm

    # AltUp correct
    hf_corrected = hf_layer.altup.correct(hf_predictions, hf_attn_ffw_laurel_gated)
    hf_first = hf_corrected[0].clone()
    if hf_layer.config.altup_correct_scale:
        hf_first = hf_layer.altup.scale_corrected_output(hf_first)

    # PLE
    hf_ple_gated = hf_layer.per_layer_input_gate(hf_first)
    hf_ple_act = hf_layer.act_fn(hf_ple_gated)
    hf_ple_mult = hf_ple_act * hf_ple_layer0
    hf_ple_proj = hf_layer.per_layer_projection(hf_ple_mult)
    hf_ple_normed = hf_layer.post_per_layer_input_norm(hf_ple_proj)

    hf_output_manual = hf_first + hf_ple_normed

    print(f"Shape: {hf_output_manual.shape}")
    print(f"Mean: {hf_output_manual.mean():.6f}")
    print(f"Std: {hf_output_manual.std():.6f}")

    print("\n[COMPARISON]")
    diff = (hf_output_actual - hf_output_manual).abs()
    print(f"layer.forward() vs manual: max={diff.max():.6e}, mean={diff.mean():.6e}")

    if diff.max() > 1e-3:
        print(f"\n⚠️  HF layer.forward() differs from manual trace!")
        print(f"   forward(): mean={hf_output_actual.mean():.6f}, std={hf_output_actual.std():.6f}")
        print(f"   manual:    mean={hf_output_manual.mean():.6f}, std={hf_output_manual.std():.6f}")
    else:
        print(f"\n✓ HF layer.forward() matches manual trace")

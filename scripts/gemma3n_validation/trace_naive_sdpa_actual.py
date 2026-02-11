#!/usr/bin/env python3
"""Capture what NaiveSDPA.forward() actually receives and produces."""

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
print("TRACE ACTUAL NaiveSDPA INPUTS AND OUTPUTS")
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

    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    # Compute HF manually to get expected output
    print("\n[COMPUTING HF MANUALLY]")
    cos, sin = position_embeddings[hf_layer0.attention_type]
    input_shape = hf_active_normed.shape[:-1]
    hidden_shape = (*input_shape, -1, hf_layer0.config.head_dim)

    q = hf_layer0.self_attn.q_proj(hf_active_normed).view(hidden_shape)
    q = hf_layer0.self_attn.q_norm(q)
    from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
    q = q.transpose(1, 2)

    k = hf_layer0.self_attn.k_proj(hf_active_normed).view(hidden_shape)
    k = hf_layer0.self_attn.k_norm(k)
    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
    k = k.transpose(1, 2)

    v = hf_layer0.self_attn.v_proj(hf_active_normed).view(hidden_shape)
    v = hf_layer0.self_attn.v_norm(v)
    v = v.transpose(1, 2)

    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    k = repeat_kv(k, 4)
    v = repeat_kv(v, 4)

    # Manual SDPA
    head_dim = q.shape[-1]
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)
    mask = torch.full((input_ids.shape[1], input_ids.shape[1]), float('-inf'), dtype=dtype, device=device)
    mask.triu_(diagonal=1)
    attn_weights = attn_weights + mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    hf_manual_attn = torch.matmul(attn_weights, v)
    hf_manual_attn = hf_manual_attn.transpose(1, 2).contiguous()
    hf_manual_out = hf_layer0.self_attn.o_proj(hf_manual_attn.view(1, input_ids.shape[1], -1))

    print(f"HF manual output [seq=3, :5]: {hf_manual_out[0, 3, :5]}")

    # Hook FS2 SDPA to capture INPUTS and compare with OUTPUTS
    print("\n[HOOKING FS2 SDPA]")
    captured = {}

    original_sdpa_forward = fs2_layer0.self_attn.sdpa.forward

    def sdpa_capture_hook(queries, queries_layout, keys, keys_layout, values, bias_cache, *, needs_weights=False):
        # Save inputs
        captured['q_input'] = queries.clone()
        captured['k_input'] = keys.clone()
        captured['v_input'] = values.clone()

        # Call ACTUAL NaiveSDPA.forward
        attns, weights = original_sdpa_forward(queries, queries_layout, keys, keys_layout, values, bias_cache, needs_weights=needs_weights)

        # Save output
        captured['sdpa_output'] = attns.clone()

        return attns, weights

    fs2_layer0.self_attn.sdpa.forward = sdpa_capture_hook

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn.sdpa.forward = original_sdpa_forward

    print(f"\n[CAPTURED FS2 SDPA INPUTS]")
    print(f"Q shape: {captured['q_input'].shape}")
    print(f"K shape: {captured['k_input'].shape}")
    print(f"V shape: {captured['v_input'].shape}")

    # Compare inputs to HF (need to convert HF to FS2 layout)
    # HF: [batch, heads, seq, dim], FS2: [batch, seq, heads, dim]
    hf_q_fs2_layout = q.transpose(1, 2)
    hf_k_fs2_layout = k.transpose(1, 2)
    hf_v_fs2_layout = v.transpose(1, 2)

    print(f"\n[COMPARING SDPA INPUTS]")
    diff_q = (captured['q_input'] - hf_q_fs2_layout).abs()
    diff_k = (captured['k_input'] - hf_k_fs2_layout).abs()
    diff_v = (captured['v_input'] - hf_v_fs2_layout).abs()

    print(f"Q diff: max={diff_q.max().item():.6e}, mean={diff_q.mean().item():.6e}")
    print(f"K diff: max={diff_k.max().item():.6e}, mean={diff_k.mean().item():.6e}")
    print(f"V diff: max={diff_v.max().item():.6e}, mean={diff_v.mean().item():.6e}")

    # Check if inputs match at the divergent position
    print(f"\n[INPUTS AT SEQ=3, HEAD=3]")
    print(f"HF Q[0,3,3,:5]:  {hf_q_fs2_layout[0, 3, 3, :5]}")
    print(f"FS2 Q[0,3,3,:5]: {captured['q_input'][0, 3, 3, :5]}")

    # Now manually run the EXACT same computation as NaiveSDPA on the captured inputs
    print(f"\n[MANUALLY RUNNING NaiveSDPA COMPUTATION ON CAPTURED INPUTS]")
    q_manual = captured['q_input'].transpose(-2, -3)  # [batch, heads, seq, dim]
    k_manual = captured['k_input'].transpose(-2, -3)
    v_manual = captured['v_input'].transpose(-2, -3)

    weights_manual = torch.matmul(q_manual, k_manual.transpose(-1, -2)) * (q_manual.size(-1) ** -0.5)

    # Get bias
    from fairseq2.models.transformer.attention_bias import maybe_get_attention_bias_tensor
    bias = maybe_get_attention_bias_tensor(
        fs2_layer0.self_attn.sdpa.bias, captured['q_input'], queries_layout, keys_layout, bias_cache
    )

    if bias is not None:
        weights_manual = weights_manual + bias

    weights_manual = torch.nn.functional.softmax(weights_manual, dim=-1, dtype=torch.float32).to(dtype)
    attns_manual = torch.matmul(weights_manual, v_manual)
    attns_manual = attns_manual.transpose(-2, -3).contiguous()

    print(f"Manual output [seq=3, head=3, :5]: {attns_manual[0, 3, 3, :5]}")
    print(f"Actual FS2 SDPA output [seq=3, head=3, :5]: {captured['sdpa_output'][0, 3, 3, :5]}")
    print(f"HF manual SDPA output [seq=3, head=3, :5]: {hf_manual_attn[0, 3, 3, :5]}")

    diff_manual = (attns_manual - captured['sdpa_output']).abs()
    print(f"\nManual vs Actual FS2 SDPA: max={diff_manual.max().item():.6e}")

    diff_hf = (hf_manual_attn - attns_manual).abs()
    print(f"HF manual vs Manual FS2: max={diff_hf.max().item():.6e}")

print("="*80)

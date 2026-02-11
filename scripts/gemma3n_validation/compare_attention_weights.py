#!/usr/bin/env python3
"""Compare attention masks and attention weights between HF and FS2."""

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
print("ATTENTION MASKS AND WEIGHTS COMPARISON")
print("="*80)

text_config = hf_model.model.language_model.config
seq_len = input_ids.shape[1]

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

    # Manually compute HF attention with attention weights
    print("\n[HF ATTENTION COMPUTATION]")

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

    # GQA expansion
    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    k = repeat_kv(k, 4)
    v = repeat_kv(v, 4)

    # Compute attention scores
    head_dim = hf_layer0.config.head_dim
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

    print(f"Raw attention scores shape: {attn_weights.shape}")  # [1, 8, 4, 4]
    print(f"Raw attention scores [head=3, seq=3, :]:")
    print(f"  {attn_weights[0, 3, 3, :]}")

    # Get HF mask - layer 0 is full_attention
    print(f"\nLayer 0 attention type: {hf_layer0.attention_type}")
    if hf_layer0.attention_type == 'sliding_attention':
        window_size = 512
        print(f"Using sliding window mask (size={window_size})")
    else:
        print(f"Using full causal mask")

    # Create mask manually
    mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
    mask.triu_(diagonal=1)  # Causal

    print(f"\nHF Mask shape: {mask.shape}")
    print(f"HF Mask:")
    print(mask)

    # Apply mask
    attn_weights_masked = attn_weights + mask
    print(f"\nAfter mask [head=3, seq=3, :]:")
    print(f"  {attn_weights_masked[0, 3, 3, :]}")

    # Softmax
    attn_weights_soft = torch.nn.functional.softmax(attn_weights_masked, dim=-1, dtype=torch.float32).to(dtype)
    print(f"\nAfter softmax [head=3, seq=3, :]:")
    print(f"  {attn_weights_soft[0, 3, 3, :]}")

    # Apply to V
    hf_attn_out = torch.matmul(attn_weights_soft, v)
    print(f"\nHF attention output [head=3, seq=3, :5]:")
    print(f"  {hf_attn_out[0, 3, 3, :5]}")

    # FS2 SDPA - capture attention weights
    print(f"\n{'='*80}")
    print("[FS2 ATTENTION COMPUTATION]")

    fs2_captured = {}

    seq_lens = [seq_len]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=seq_len)
    attn_bias_cache = AttentionBiasCache()

    # Hook NaiveSDPA to capture attention weights
    original_forward = fs2_layer0.self_attn.sdpa.forward

    def sdpa_hook(queries, queries_layout, keys, keys_layout, values, bias_cache, *, needs_weights=False):
        # Manually compute to capture intermediates
        q = queries.transpose(-2, -3)  # [batch, heads, seq, dim]
        k = keys.transpose(-2, -3)
        v = values.transpose(-2, -3)

        # Scores (use head_dim for scaling)
        head_dim = q.shape[-1]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

        print(f"FS2 Raw attention scores shape: {attn_weights.shape}")
        print(f"FS2 Raw attention scores [head=3, seq=3, :]:")
        print(f"  {attn_weights[0, 3, 3, :]}")

        # Get bias/mask from cache using the correct API
        from fairseq2.models.transformer.attention_bias import maybe_get_attention_bias_tensor
        bias = maybe_get_attention_bias_tensor(
            fs2_layer0.self_attn.sdpa.bias, queries, queries_layout, keys_layout, bias_cache
        )

        print(f"\nFS2 Bias shape: {bias.shape if bias is not None else 'None'}")
        if bias is not None:
            print(f"FS2 Bias:")
            # Bias might be [batch, heads, seq, seq] or [heads, seq, seq] or [seq, seq]
            if bias.dim() == 4:
                print(bias[0, 0, :, :])
            elif bias.dim() == 3:
                print(bias[0, :, :])
            else:
                print(bias)

        # Apply bias
        if bias is not None:
            attn_weights = attn_weights + bias
        print(f"\nFS2 After bias [head=3, seq=3, :]:")
        print(f"  {attn_weights[0, 3, 3, :]}")

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        print(f"\nFS2 After softmax [head=3, seq=3, :]:")
        print(f"  {attn_weights[0, 3, 3, :]}")

        # Apply to V
        attns = torch.matmul(attn_weights, v)
        attns = attns.transpose(-2, -3).contiguous()  # Back to [batch, seq, heads, dim]

        print(f"\nFS2 attention output [head=3, seq=3, :5]:")
        print(f"  {attns[0, 3, 3, :5]}")

        return attns, None

    fs2_layer0.self_attn.sdpa.forward = sdpa_hook

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn.sdpa.forward = original_forward

print("="*80)

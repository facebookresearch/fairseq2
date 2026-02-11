#!/usr/bin/env python3
"""Check Q, K, V at the exact sequence position where SDPA diverges."""

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
print("Q, K, V AT DIVERGENT POSITION (seq=3, head=3)")
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

    # Hook HF to capture Q, K, V
    hf_captured = {}

    original_hf_forward = hf_layer0.self_attn.__class__.forward
    def hf_hook(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        cos, sin = position_embeddings
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        q = self.q_norm(q)
        from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)

        k = self.k_proj(hidden_states).view(hidden_shape)
        k = self.k_norm(k)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden_states).view(hidden_shape)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        # GQA expansion
        def repeat_kv(hidden_states, n_rep):
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

        k_repeated = repeat_kv(k, self.config.num_attention_heads // self.config.num_key_value_heads)
        v_repeated = repeat_kv(v, self.config.num_attention_heads // self.config.num_key_value_heads)

        hf_captured['q'] = q.clone()
        hf_captured['k'] = k_repeated.clone()
        hf_captured['v'] = v_repeated.clone()

        return original_hf_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)

    hf_layer0.self_attn.__class__.forward = hf_hook
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    hf_layer0.self_attn.__class__.forward = original_hf_forward

    # Hook FS2 SDPA to capture Q, K, V
    fs2_captured = {}

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    original_sdpa_forward = fs2_layer0.self_attn.sdpa.forward
    def sdpa_hook(q, q_layout, k, k_layout, v, bias_cache, needs_weights=False):
        fs2_captured['q'] = q.clone()
        fs2_captured['k'] = k.clone()
        fs2_captured['v'] = v.clone()

        # Also run SDPA and capture output
        attns, weights = original_sdpa_forward(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)
        fs2_captured['sdpa_output'] = attns.clone()
        return attns, weights

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

    fs2_layer0.self_attn.sdpa.forward = original_sdpa_forward

    # Convert FS2 to HF layout
    fs2_q = fs2_captured['q'].transpose(1, 2)
    fs2_k = fs2_captured['k'].transpose(1, 2)
    fs2_v = fs2_captured['v'].transpose(1, 2)

    # Check ALL sequence positions for head 3
    seq_len = input_ids.shape[1]
    print(f"\nSequence length: {seq_len}")
    print(f"\n{'='*80}")
    print(f"HEAD 3 Q, K, V ACROSS ALL SEQUENCE POSITIONS")
    print(f"{'='*80}")

    for seq_pos in range(seq_len):
        print(f"\n[SEQ={seq_pos}, HEAD=3]")

        # Q at this position
        hf_q = hf_captured['q'][0, 3, seq_pos, :5]
        fs2_q_val = fs2_q[0, 3, seq_pos, :5]
        diff_q = (hf_q - fs2_q_val).abs()

        print(f"  Q HF:  {hf_q}")
        print(f"  Q FS2: {fs2_q_val}")
        print(f"  Q diff: max={diff_q.max().item():.6e}")

        # K at this position (for attention scores with seq_pos)
        # In attention, Q[seq_pos] attends to all K positions
        # Let's check if K at this position differs
        hf_k = hf_captured['k'][0, 3, seq_pos, :5]
        fs2_k_val = fs2_k[0, 3, seq_pos, :5]
        diff_k = (hf_k - fs2_k_val).abs()

        print(f"  K HF:  {hf_k}")
        print(f"  K FS2: {fs2_k_val}")
        print(f"  K diff: max={diff_k.max().item():.6e}")

        # V at this position
        hf_v = hf_captured['v'][0, 3, seq_pos, :5]
        fs2_v_val = fs2_v[0, 3, seq_pos, :5]
        diff_v = (hf_v - fs2_v_val).abs()

        print(f"  V HF:  {hf_v}")
        print(f"  V FS2: {fs2_v_val}")
        print(f"  V diff: max={diff_v.max().item():.6e}")

        if diff_q.max().item() > 1e-5 or diff_k.max().item() > 1e-5 or diff_v.max().item() > 1e-5:
            print(f"  ⚠️  DIVERGES!")

    # Now check SDPA output at the divergent position
    print(f"\n{'='*80}")
    print(f"SDPA OUTPUT AT DIVERGENT POSITION")
    print(f"{'='*80}")

    # Capture HF SDPA output using hook
    hf_sdpa_captured = {}

    import torch.nn.functional as F
    original_torch_sdpa = F.scaled_dot_product_attention

    def torch_sdpa_hook(*args, **kwargs):
        output = original_torch_sdpa(*args, **kwargs)
        hf_sdpa_captured['output'] = output.clone()
        return output

    F.scaled_dot_product_attention = torch_sdpa_hook

    # Re-run HF attention
    hf_attn_out2, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )

    F.scaled_dot_product_attention = original_torch_sdpa

    # Compare SDPA outputs at seq=3, head=3
    hf_sdpa_out = hf_sdpa_captured['output'].transpose(1, 2)  # [batch, seq, heads, dim]
    fs2_sdpa_out = fs2_captured['sdpa_output']

    print(f"\n[SDPA OUTPUT at seq=3, head=3, first 10 dims]")
    print(f"HF:  {hf_sdpa_out[0, 3, 3, :10]}")
    print(f"FS2: {fs2_sdpa_out[0, 3, 3, :10]}")
    diff = (hf_sdpa_out[0, 3, 3, :] - fs2_sdpa_out[0, 3, 3, :]).abs()
    print(f"Diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)

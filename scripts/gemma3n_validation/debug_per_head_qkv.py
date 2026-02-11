#!/usr/bin/env python3
"""Compare Q, K, V values for EACH head to identify which heads diverge."""

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
print("PER-HEAD Q, K, V COMPARISON")
print("="*80)

text_config = hf_model.model.language_model.config
num_q_heads = text_config.num_attention_heads  # 8
num_kv_heads = text_config.num_key_value_heads  # 2

print(f"\nArchitecture: {num_q_heads} Q heads, {num_kv_heads} KV heads")
print(f"GQA ratio: {num_q_heads // num_kv_heads}:1")

with torch.no_grad():
    # Setup (same as before)
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

    # Hook HF attention to capture Q, K, V AFTER GQA expansion
    hf_captured = {}

    original_hf_forward = hf_layer0.self_attn.__class__.forward
    def hf_hook(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        # Manually compute Q, K, V
        cos, sin = position_embeddings
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        q = self.q_norm(q)
        from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]

        k = self.k_proj(hidden_states).view(hidden_shape)
        k = self.k_norm(k)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        k = k.transpose(1, 2)  # [batch, kv_heads, seq, dim]

        v = self.v_proj(hidden_states).view(hidden_shape)
        v = self.v_norm(v)
        v = v.transpose(1, 2)  # [batch, kv_heads, seq, dim]

        # GQA expansion (repeat K/V)
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

        # Continue with normal forward
        return original_hf_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)

    hf_layer0.self_attn.__class__.forward = hf_hook
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    hf_layer0.self_attn.__class__.forward = original_hf_forward

    # Hook FS2 SDPA to capture Q, K, V right before SDPA call
    fs2_captured = {}

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    # Hook SDPA to capture inputs
    original_sdpa_forward = fs2_layer0.self_attn.sdpa.forward
    def sdpa_hook(q, q_layout, k, k_layout, v, bias_cache, needs_weights=False):
        # Capture Q, K, V right before SDPA computation
        fs2_captured['q'] = q.clone()
        fs2_captured['k'] = k.clone()  # Already after GQA expansion
        fs2_captured['v'] = v.clone()
        return original_sdpa_forward(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)

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

    print(f"\n[CAPTURED SHAPES]")
    print(f"HF Q: {hf_captured['q'].shape}")  # [batch, 8, seq, dim]
    print(f"HF K (after GQA): {hf_captured['k'].shape}")  # [batch, 8, seq, dim]
    print(f"HF V (after GQA): {hf_captured['v'].shape}")  # [batch, 8, seq, dim]
    print(f"FS2 Q: {fs2_captured['q'].shape}")  # [batch, seq, 8, dim]
    print(f"FS2 K (after GQA): {fs2_captured['k'].shape}")  # [batch, seq, 8, dim]
    print(f"FS2 V (after GQA): {fs2_captured['v'].shape}")  # [batch, seq, 8, dim]

    # Convert FS2 to HF layout for comparison
    fs2_q = fs2_captured['q'].transpose(1, 2)  # [batch, heads, seq, dim]
    fs2_k = fs2_captured['k'].transpose(1, 2)
    fs2_v = fs2_captured['v'].transpose(1, 2)

    # Compare PER HEAD
    print(f"\n{'='*80}")
    print(f"PER-HEAD COMPARISON (batch=0, seq=0, first 5 dims)")
    print(f"{'='*80}")

    for head_idx in range(num_q_heads):
        print(f"\n[Q HEAD {head_idx}]")
        hf_q_head = hf_captured['q'][0, head_idx, 0, :5]
        fs2_q_head = fs2_q[0, head_idx, 0, :5]
        diff = (hf_q_head - fs2_q_head).abs()
        print(f"  HF:  {hf_q_head}")
        print(f"  FS2: {fs2_q_head}")
        print(f"  Diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

        # Which KV head does this Q head use in GQA?
        kv_head_idx = head_idx // (num_q_heads // num_kv_heads)
        print(f"  -> Uses KV head {kv_head_idx}")

        hf_k_head = hf_captured['k'][0, head_idx, 0, :5]
        fs2_k_head = fs2_k[0, head_idx, 0, :5]
        diff_k = (hf_k_head - fs2_k_head).abs()
        print(f"  K HF:  {hf_k_head}")
        print(f"  K FS2: {fs2_k_head}")
        print(f"  K Diff: max={diff_k.max().item():.6e}")

        hf_v_head = hf_captured['v'][0, head_idx, 0, :5]
        fs2_v_head = fs2_v[0, head_idx, 0, :5]
        diff_v = (hf_v_head - fs2_v_head).abs()
        print(f"  V HF:  {hf_v_head}")
        print(f"  V FS2: {fs2_v_head}")
        print(f"  V Diff: max={diff_v.max().item():.6e}")

        if diff.max().item() > 1e-5 or diff_k.max().item() > 1e-5 or diff_v.max().item() > 1e-5:
            print(f"  ⚠️  DIVERGES")
        else:
            print(f"  ✅ MATCHES")

print("="*80)

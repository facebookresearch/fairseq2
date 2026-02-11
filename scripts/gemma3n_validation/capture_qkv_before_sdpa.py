#!/usr/bin/env python3
"""Capture actual Q, K, V tensors right before SDPA and compare."""

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
print("CAPTURE ACTUAL Q, K, V BEFORE SDPA")
print("="*80)

text_config = hf_model.model.language_model.config

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

    # Hook HF attention to capture Q, K, V before SDPA
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
        q = q.transpose(1, 2)

        k = self.k_proj(hidden_states).view(hidden_shape)
        k = self.k_norm(k)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden_states).view(hidden_shape)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        hf_captured['q'] = q.clone()
        hf_captured['k'] = k.clone()
        hf_captured['v'] = v.clone()

        # Continue with normal forward
        return original_hf_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)

    hf_layer0.self_attn.__class__.forward = hf_hook
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    hf_layer0.self_attn.__class__.forward = original_hf_forward

    # Hook FS2 attention to capture Q, K, V BEFORE GQA expansion
    fs2_captured = {}

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    # Hook the internal _project_kv method to capture before repeat
    original_project_kv = fs2_layer0.self_attn._project_kv
    def project_kv_hook(keys, keys_layout, values, state_bag=None):
        k, v = original_project_kv(keys, keys_layout, values, state_bag)
        # k, v are still [batch, seq, kv_heads, dim] here
        fs2_captured['k_before_repeat'] = k.clone()
        fs2_captured['v_before_repeat'] = v.clone()
        return k, v

    fs2_layer0.self_attn._project_kv = project_kv_hook

    original_project_q = fs2_layer0.self_attn._project_q
    def project_q_hook(seqs, seqs_layout, state_bag=None):
        q = original_project_q(seqs, seqs_layout, state_bag)
        # q is [batch, seq, heads, dim]
        fs2_captured['q'] = q.clone()
        return q

    fs2_layer0.self_attn._project_q = project_q_hook

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn._project_kv = original_project_kv
    fs2_layer0.self_attn._project_q = original_project_q

    print(f"\n[HF Q, K, V shapes]")
    print(f"Q: {hf_captured['q'].shape}")  # [batch, heads, seq, dim]
    print(f"K: {hf_captured['k'].shape}")  # [batch, kv_heads, seq, dim]
    print(f"V: {hf_captured['v'].shape}")

    print(f"\n[FS2 Q, K, V shapes (before GQA expansion)]")
    print(f"Q: {fs2_captured['q'].shape}")  # [batch, seq, heads, dim]
    print(f"K: {fs2_captured['k_before_repeat'].shape}")  # [batch, seq, kv_heads, dim]
    print(f"V: {fs2_captured['v_before_repeat'].shape}")

    # Compare (need to transpose FS2 to match HF format)
    fs2_q = fs2_captured['q'].transpose(1, 2)  # -> [batch, heads, seq, dim]
    fs2_k = fs2_captured['k_before_repeat'].transpose(1, 2)  # -> [batch, kv_heads, seq, dim]
    fs2_v = fs2_captured['v_before_repeat'].transpose(1, 2)

    print(f"\n[COMPARING Q, K, V]")
    diff_q = (hf_captured['q'] - fs2_q).abs()
    diff_k = (hf_captured['k'] - fs2_k).abs()
    diff_v = (hf_captured['v'] - fs2_v).abs()

    print(f"Q diff: max={diff_q.max().item():.6e}, mean={diff_q.mean().item():.6e}")
    print(f"K diff: max={diff_k.max().item():.6e}, mean={diff_k.mean().item():.6e}")
    print(f"V diff: max={diff_v.max().item():.6e}, mean={diff_v.mean().item():.6e}")

    # Show sample values
    print(f"\n[Q SAMPLE VALUES - position 0, head 0]")
    print(f"HF Q[0,0,0,:5]:  {hf_captured['q'][0,0,0,:5]}")
    print(f"FS2 Q[0,0,0,:5]: {fs2_q[0,0,0,:5]}")

    print(f"\n[K SAMPLE VALUES - position 0, head 0]")
    print(f"HF K[0,0,0,:5]:  {hf_captured['k'][0,0,0,:5]}")
    print(f"FS2 K[0,0,0,:5]: {fs2_k[0,0,0,:5]}")

    print(f"\n[ATTENTION OUTPUT COMPARISON]")
    diff_attn = (hf_attn_out - fs2_attn_out).abs()
    print(f"Attention output diff: max={diff_attn.max().item():.6e}, mean={diff_attn.mean().item():.6e}")

print("="*80)

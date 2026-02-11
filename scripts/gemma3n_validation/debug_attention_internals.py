#!/usr/bin/env python3
"""Debug attention internals to find divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("ATTENTION INTERNALS DEBUG")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup inputs (same as before)
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

    # Get layer 0
    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    # Get active prediction (already normalized)
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    print(f"\n[INPUT TO ATTENTION]")
    print(f"Shape: {hf_active_normed.shape}")
    diff = (hf_active_normed - fs2_active_normed).abs()
    print(f"Input diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Extract attention modules
    hf_attn = hf_layer0.self_attn
    fs2_attn = fs2_layer0.self_attn

    print(f"\n[ATTENTION CONFIG]")
    print(f"HF num_heads: {hf_attn.config.num_attention_heads}, num_kv_heads: {hf_attn.config.num_key_value_heads}")
    print(f"FS2 num_heads: {fs2_attn.num_heads}, num_key_value_heads: {fs2_attn.num_key_value_heads}")
    print(f"HF head_dim: {hf_attn.config.head_dim}")
    print(f"FS2 head_dim: {fs2_attn.head_dim}")

    # Compare Q projection weights
    print(f"\n[Q PROJECTION WEIGHTS]")
    hf_q_weight = hf_attn.q_proj.weight
    fs2_q_weight = fs2_attn.q_proj.weight
    diff = (hf_q_weight - fs2_q_weight).abs()
    print(f"  Weight diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Compare Q projections
    print(f"\n[Q PROJECTION OUTPUT]")
    hf_q = hf_attn.q_proj(hf_active_normed)
    fs2_q = fs2_attn.q_proj(fs2_active_normed)
    print(f"  HF Q shape: {hf_q.shape}")
    print(f"  FS2 Q shape: {fs2_q.shape}")
    diff = (hf_q - fs2_q).abs()
    print(f"  Q diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Reshape for multi-head
    bsz, q_len, _ = hf_active_normed.shape
    num_heads = hf_attn.config.num_attention_heads
    head_dim = hf_attn.config.head_dim

    # FS2 format: [batch, seq_len, num_heads, head_dim] (before transpose)
    fs2_q_reshaped = fs2_q.view(bsz, q_len, fs2_attn.num_heads, fs2_attn.head_dim)

    # HF format: [batch, num_heads, seq_len, head_dim] (after transpose)
    hf_q_reshaped = hf_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    print(f"\n[Q RESHAPED]")
    print(f"  HF shape (transposed): {hf_q_reshaped.shape}")
    print(f"  FS2 shape (pre-transpose): {fs2_q_reshaped.shape}")

    # Apply Q normalization (FS2 applies before transpose, HF after)
    print(f"\n[Q NORMALIZATION]")
    fs2_q_normed = fs2_attn.q_norm(fs2_q_reshaped)  # [batch, seq, heads, head_dim]
    hf_q_normed = hf_attn.q_norm(hf_q_reshaped)     # [batch, heads, seq, head_dim]

    print(f"  HF Q normed shape: {hf_q_normed.shape}")
    print(f"  FS2 Q normed shape: {fs2_q_normed.shape}")

    # Compare (need to transpose one to match)
    hf_q_normed_transposed = hf_q_normed.transpose(1, 2)  # -> [batch, seq, heads, head_dim]
    diff = (hf_q_normed_transposed - fs2_q_normed).abs()
    print(f"  Q normed diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Apply RoPE to Q
    print(f"\n[ROPE APPLICATION TO Q]")
    hf_cos, hf_sin = position_embeddings[hf_layer0.attention_type]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # HF applies to [batch, heads, seq, head_dim]
    hf_q_with_rope = (hf_q_normed * hf_cos) + (rotate_half(hf_q_normed) * hf_sin)
    print(f"  HF Q with RoPE shape: {hf_q_with_rope.shape}")

    # FS2 applies to [batch, seq, heads, head_dim]
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    print(f"  FS2 pos_encoder theta: {fs2_attn.pos_encoder.theta}")
    print(f"  FS2 pos_encoder encoding_dim: {fs2_attn.pos_encoder.encoding_dim}")

    fs2_q_with_rope = fs2_attn.pos_encoder(fs2_q_normed, batch_layout, state_bag=state_bag)
    print(f"  FS2 Q with RoPE shape: {fs2_q_with_rope.shape}")

    # Compare (transpose HF to match FS2 format)
    hf_q_with_rope_transposed = hf_q_with_rope.transpose(1, 2)  # -> [batch, seq, heads, head_dim]
    diff = (hf_q_with_rope_transposed - fs2_q_with_rope).abs()
    print(f"  Q with RoPE diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)

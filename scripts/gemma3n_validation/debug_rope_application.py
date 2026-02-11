#!/usr/bin/env python3
"""Debug RoPE computation in layer 0 attention."""

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
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("ROPE COMPUTATION DEBUG")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Get layer 0
    hf_layer0 = hf_model.model.language_model.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    print(f"\n[LAYER 0 INFO]")
    print(f"HF attention type: {hf_layer0.attention_type}")
    print(f"FS2 has pos_encoder: {fs2_layer0.self_attn.pos_encoder is not None}")
    if fs2_layer0.self_attn.pos_encoder is not None:
        print(f"FS2 pos_encoder type: {type(fs2_layer0.self_attn.pos_encoder)}")
        print(f"FS2 RoPE theta: {fs2_layer0.self_attn.pos_encoder.theta}")
        print(f"FS2 RoPE encoding_dim: {fs2_layer0.self_attn.pos_encoder.encoding_dim}")

    # Build test input
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

    # HF RoPE computation
    print(f"\n[HF ROPE COMPUTATION]")
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    print(f"Position IDs: {position_ids}")

    hf_cos, hf_sin = hf_lm.rotary_emb(hf_hidden_4d, position_ids, hf_layer0.attention_type)
    print(f"HF RoPE cos shape: {hf_cos.shape}")
    print(f"HF RoPE sin shape: {hf_sin.shape}")
    print(f"HF cos[0, 0, :5]: {hf_cos[0, 0, :5]}")
    print(f"HF sin[0, 0, :5]: {hf_sin[0, 0, :5]}")

    # FS2 RoPE computation (extract what the pos_encoder would compute)
    print(f"\n[FS2 ROPE COMPUTATION]")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    # Extract a test query to see what RoPE does
    test_q = hf_hidden_4d[0]  # Use first prediction
    print(f"Test query shape: {test_q.shape}")

    # Apply FS2 RoPE encoder
    pos_encoder = fs2_layer0.self_attn.pos_encoder
    test_q_with_rope = pos_encoder(test_q, batch_layout, state_bag=state_bag)

    print(f"After RoPE shape: {test_q_with_rope.shape}")

    # Compare RoPE application
    print(f"\n[COMPARING ROPE APPLICATION]")
    print(f"Original query: {test_q[0, 0, :5]}")
    print(f"After FS2 RoPE: {test_q_with_rope[0, 0, :5]}")

    # Try to manually apply HF RoPE to same query
    # HF applies: q_embed = (q * cos) + (rotate_half(q) * sin)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    test_q_with_hf_rope = (test_q * hf_cos) + (rotate_half(test_q) * hf_sin)
    print(f"After HF RoPE:  {test_q_with_hf_rope[0, 0, :5]}")

    diff = (test_q_with_hf_rope - test_q_with_rope).abs()
    print(f"\nRoPE application diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)

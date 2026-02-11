#!/usr/bin/env python3
"""Compare RoPE computation between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


device = torch.device("cpu")

# Load models
print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True,
)
hf_model.eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
fs2_model.eval()

hf_state_dict = hf_model.state_dict()
fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
fs2_model.load_state_dict(fs2_state_dict, strict=False)

print("="*80)
print("RoPE Configuration")
print("="*80)

# Check HF RoPE config
hf_lm = hf_model.model.language_model
print(f"HF config:")
print(f"  head_dim: {hf_lm.config.head_dim}")
print(f"  layer_types[0]: {hf_lm.config.layer_types[0]}")
slide_params = hf_lm.config.rope_parameters.get("sliding_attention", {})
print(f"  sliding_attention rope_theta: {slide_params.get('rope_theta', 'N/A')}")

# Check FS2 RoPE
fs2_layer0 = fs2_model.decoder.layers[0]
print(f"\nFS2 config:")
print(f"  rope_theta: {config.rope_theta}")
print(f"  rope_theta_global: {config.rope_theta_global}")
print(f"  head_dim: {config.head_dim}")
print(f"  pos_encoder type: {type(fs2_layer0.self_attn.pos_encoder).__name__}")

if hasattr(fs2_layer0.self_attn.pos_encoder, 'theta'):
    print(f"  pos_encoder theta: {fs2_layer0.self_attn.pos_encoder.theta}")
if hasattr(fs2_layer0.self_attn.pos_encoder, 'dual_theta'):
    print(f"  pos_encoder dual_theta: {fs2_layer0.self_attn.pos_encoder.dual_theta}")

print("\n" + "="*80)
print("RoPE Application Test")
print("="*80)

# Create test tensor (same as Q/K shape before RoPE)
seq_len = 3
test_tensor = torch.randn(1, seq_len, 2, 256, device=device, dtype=torch.float32)  # B, S, H, D
print(f"Test tensor shape: {test_tensor.shape} (B, S, H, D)")

# HF RoPE
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding, apply_rotary_pos_emb
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
cos, sin = rope(test_tensor, position_ids, "sliding_attention")

print(f"\nHF RoPE cos/sin shapes: {cos.shape}, {sin.shape}")
print(f"HF cos sample [0,0,:5]: {cos[0,0,:5]}")
print(f"HF sin sample [0,0,:5]: {sin[0,0,:5]}")

hf_output = apply_rotary_pos_emb(test_tensor, cos, sin, unsqueeze_dim=2)
print(f"HF output: mean={hf_output.mean():.6f}, std={hf_output.std():.6f}")

# FS2 RoPE
batch_layout = BatchLayout(torch.Size([1, seq_len]), [seq_len], device=device)
state_bag = IncrementalStateBag(max_num_steps=seq_len)

# FS2 expects (B, S, H, D) and applies RoPE
fs2_output = fs2_layer0.self_attn.pos_encoder(test_tensor, batch_layout, state_bag=state_bag)
print(f"\nFS2 output: mean={fs2_output.mean():.6f}, std={fs2_output.std():.6f}")

# Compare
diff = (hf_output - fs2_output).abs()
print(f"\nRoPE diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

if diff.max() < 1e-5:
    print("✓ RoPE outputs match!")
else:
    print("❌ RoPE outputs differ significantly")
    print("\nThis explains the Q/K difference!")
    print("\nLet's check if DualRotaryEncoder is the right choice...")
    print(f"DualRotaryEncoder splits head_dim in half: {config.head_dim} → {config.head_dim//2} + {config.head_dim//2}")
    print(f"First half uses theta={config.rope_theta}")
    print(f"Second half uses dual_theta={config.rope_theta_global}")
    print("\nHF might use a different approach for sliding_attention layers.")

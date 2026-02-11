#!/usr/bin/env python3
"""Check if embedding weights are pre-scaled in HF checkpoint."""

import torch
from transformers import AutoModelForCausalLM
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

print("="*80)
print("EMBEDDING WEIGHTS COMPARISON")
print("="*80)

# Compare raw embedding weights
hf_embed_weight = hf_model.model.language_model.embed_tokens.weight
fs2_embed_weight = fs2_model.decoder_frontend.embed.weight

print(f"\n[RAW EMBEDDING WEIGHTS]")
print(f"HF shape:  {hf_embed_weight.shape}")
print(f"FS2 shape: {fs2_embed_weight.shape}")

diff = (hf_embed_weight - fs2_embed_weight).abs()
print(f"Max diff:  {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
print(f"Weights match: {torch.allclose(hf_embed_weight, fs2_embed_weight, atol=1e-6)}")

# Check scaling factor
print(f"\n[SCALING FACTORS]")
print(f"model_dim = {config.model_dim}")
print(f"Expected scale = sqrt({config.model_dim}) = {config.model_dim**0.5:.6f}")
print(f"FS2 frontend scale: {fs2_model.decoder_frontend.scale:.6f}")

# Test embedding lookup
print(f"\n[EMBEDDING LOOKUP TEST]")
test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

# HF embed_tokens forward
hf_embeds_raw = hf_model.model.language_model.embed_tokens(test_ids)
print(f"HF embeds (from embed_tokens): shape={hf_embeds_raw.shape}, mean={hf_embeds_raw.mean().item():.6f}, std={hf_embeds_raw.std().item():.6f}")

# FS2 embed forward (raw, no scaling)
fs2_embeds_raw = fs2_model.decoder_frontend.embed(test_ids)
print(f"FS2 embeds (raw):  shape={fs2_embeds_raw.shape}, mean={fs2_embeds_raw.mean().item():.6f}, std={fs2_embeds_raw.std().item():.6f}")

# FS2 after scaling
fs2_embeds_scaled = fs2_embeds_raw * fs2_model.decoder_frontend.scale
print(f"FS2 embeds (scaled): shape={fs2_embeds_scaled.shape}, mean={fs2_embeds_scaled.mean().item():.6f}, std={fs2_embeds_scaled.std().item():.6f}")

# Compare
diff_raw = (hf_embeds_raw - fs2_embeds_raw).abs()
print(f"\nHF vs FS2 raw: max diff = {diff_raw.max().item():.6e}, mean diff = {diff_raw.mean().item():.6e}")

diff_scaled = (hf_embeds_raw - fs2_embeds_scaled).abs()
print(f"HF vs FS2 scaled: max diff = {diff_scaled.max().item():.6e}, mean diff = {diff_scaled.mean().item():.6e}")

# Check if HF applies scaling
print(f"\n[HF EMBED_TOKENS CLASS]")
print(f"Type: {type(hf_model.model.language_model.embed_tokens)}")
print(f"Has embed_scale: {hasattr(hf_model.model.language_model.embed_tokens, 'embed_scale')}")
if hasattr(hf_model.model.language_model.embed_tokens, 'embed_scale'):
    print(f"embed_scale value: {hf_model.model.language_model.embed_tokens.embed_scale}")

print("="*80)

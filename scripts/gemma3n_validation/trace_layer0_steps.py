#!/usr/bin/env python3
"""Step-by-step comparison through layer 0 to find divergence point."""

import torch
import math
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

# Disable sparsity
for layer in hf_model.model.language_model.layers:
    if hasattr(layer, 'mlp'):
        layer.mlp.activation_sparsity = 0.0

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("STEP-BY-STEP LAYER 0 COMPARISON")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup inputs
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    hf_per_layer_inputs = hf_lm.get_per_layer_inputs(input_ids)
    hf_per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, hf_per_layer_inputs)

    # Build 4D stack
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

    # RoPE embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # Get layers
    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    per_layer_input_0 = hf_per_layer_inputs[:, :, 0, :]

    # FS2 setup
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    print(f"\n[STEP 1: AltUp Predict]")
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)  # FS2 altup.forward does predict

    for i in range(4):
        diff = (hf_predictions[i] - fs2_predictions[i]).abs()
        print(f"  Prediction [{i}]: max diff = {diff.max().item():.6e}")

    print(f"\n[STEP 2: Extract Active Prediction]")
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    diff = (hf_active - fs2_active).abs()
    print(f"  Max diff = {diff.max().item():.6e}")

    print(f"\n[STEP 3: Input Layernorm]")
    hf_active_normed = hf_layer0.input_layernorm(hf_active)
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)
    diff = (hf_active_normed - fs2_active_normed).abs()
    print(f"  Max diff = {diff.max().item():.6e}")

    print(f"\n[STEP 4: LAuReL]")
    hf_laurel = hf_layer0.laurel(hf_active_normed)
    fs2_laurel = fs2_layer0.laurel(fs2_active_normed)
    diff = (hf_laurel - fs2_laurel).abs()
    print(f"  Max diff = {diff.max().item():.6e}")

    print(f"\n[STEP 5: Self-Attention]")
    hf_attn, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    fs2_attn = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )
    diff = (hf_attn - fs2_attn).abs()
    print(f"  Max diff = {diff.max().item():.6e}")
    print(f"  Mean diff = {diff.mean().item():.6e}")

print("="*80)

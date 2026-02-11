#!/usr/bin/env python3
"""Isolate layer 0 by feeding identical inputs to HF and FS2."""

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
print("LAYER 0 ISOLATED TEST - IDENTICAL INPUTS")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup identical inputs
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    hf_per_layer_inputs = hf_lm.get_per_layer_inputs(input_ids)
    hf_per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, hf_per_layer_inputs)

    # Build 4D stack (HF way)
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

    # Compute RoPE position embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # Get layer 0 objects
    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    print(f"\n[INPUTS TO LAYER 0]")
    print(f"Hidden states shape: {hf_hidden_4d.shape}")
    print(f"PLE shape: {hf_per_layer_inputs.shape}")
    print(f"Layer 0 attention type: {hf_layer0.attention_type}")

    # Extract PLE for layer 0
    per_layer_input_0 = hf_per_layer_inputs[:, :, 0, :]
    print(f"PLE for layer 0 shape: {per_layer_input_0.shape}")

    # HF layer 0 forward
    print(f"\n[HF LAYER 0 FORWARD]")
    hf_output = hf_layer0(
        hf_hidden_4d,
        position_embeddings[hf_layer0.attention_type],
        per_layer_input_0,
    )
    print(f"Output shape: {hf_output.shape}")
    print(f"Output mean: {hf_output.mean().item():.6f}, std: {hf_output.std().item():.6f}")

    # FS2 layer 0 forward with same inputs
    print(f"\n[FS2 LAYER 0 FORWARD]")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    # Use the SAME 4D hidden states and PLE as HF
    fs2_output = fs2_layer0(
        hf_hidden_4d,  # Use HF's 4D stack
        batch_layout,
        attn_bias_cache,
        per_layer_input=per_layer_input_0,  # Use HF's PLE
        state_bag=state_bag,
    )
    print(f"Output shape: {fs2_output.shape}")
    print(f"Output mean: {fs2_output.mean().item():.6f}, std: {fs2_output.std().item():.6f}")

    # Compare outputs
    print(f"\n[COMPARING LAYER 0 OUTPUTS]")
    for i in range(4):
        diff = (hf_output[i] - fs2_output[i]).abs()
        print(f"  Prediction [{i}]: max diff = {diff.max().item():.6e}, mean diff = {diff.mean().item():.6e}")

        # Sample values
        print(f"    HF  sample: {hf_output[i, 0, 0, :5].tolist()}")
        print(f"    FS2 sample: {fs2_output[i, 0, 0, :5].tolist()}")

print("="*80)

#!/usr/bin/env python3
"""Debug the final unstack operation to find where divergence happens."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

# Disable HF activation sparsity
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
print("DEBUGGING FINAL UNSTACK OPERATION")
print("="*80)

# Debug: print config layer types
text_config = hf_model.model.language_model.config
print(f"Config layer_types (unique): {set(text_config.layer_types)}")
print(f"RoPE parameters keys: {list(hf_model.model.language_model.rotary_emb.rope_type.keys())}")
print("="*80)

with torch.no_grad():
    # HF forward - capture 4D hidden state after all layers
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    per_layer_inputs = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs)

    # Build 4D stack (same as HF)
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

    # Compute position embeddings (RoPE cos/sin) for each layer type
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # Forward through all 30 HF layers
    for decoder_layer in hf_lm.layers:
        per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]
        hf_hidden_4d = decoder_layer(
            hf_hidden_4d,
            position_embeddings[decoder_layer.attention_type],  # Pass correct (cos, sin) for this layer
            per_layer_input,
        )

    print(f"\n[AFTER ALL 30 LAYERS - 4D HIDDEN STATE]")
    print(f"  HF hidden_4d shape: {hf_hidden_4d.shape}")

    # HF unstack operation
    target_magnitude = torch.mean(hf_hidden_4d[0] ** 2, dim=-1, keepdim=True) ** 0.5
    temp_hidden_states = [hf_hidden_4d[0]]
    for i in range(1, 4):
        altup_unemb_proj = hf_lm.altup_unembed_projections[i - 1](hf_hidden_4d[i])
        new_magnitude = torch.mean(altup_unemb_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_unemb_proj = altup_unemb_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_unemb_proj)

    hf_unstacked = torch.stack(temp_hidden_states)
    hf_averaged = torch.mean(hf_unstacked, dim=0)
    hf_final = hf_lm.norm(hf_averaged)
    hf_logits = hf_model.lm_head(hf_final)

    # FS2 forward - capture 4D hidden state before unstack
    captured = {}

    original_unstack = fs2_model.decoder._unstack_altup
    def hook_unstack(hidden_states):
        captured['fs2_4d'] = hidden_states.clone()
        return original_unstack(hidden_states)

    fs2_model.decoder._unstack_altup = hook_unstack

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    fs2_logits = fs2_model(input_ids, batch_layout)

    fs2_model.decoder._unstack_altup = original_unstack

    captured_fs2_4d = captured['fs2_4d']
    print(f"  FS2 hidden_4d shape: {captured_fs2_4d.shape}")

    # Compare 4D hidden states before unstack
    print(f"\n[COMPARING 4D HIDDEN STATES BEFORE UNSTACK]")
    for i in range(4):
        diff = (hf_hidden_4d[i] - captured_fs2_4d[i]).abs()
        print(f"  prediction[{i}]: max diff = {diff.max().item():.6e}, mean diff = {diff.mean().item():.6e}")

    print(f"\n[COMPARING FINAL OUTPUTS]")
    print(f"  HF logits shape:  {hf_logits.shape}")
    print(f"  FS2 logits shape: {fs2_logits.shape}")

    diff = (hf_logits - fs2_logits).abs()
    print(f"  Max diff:  {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")

    # Check token predictions
    hf_predicted = hf_logits.argmax(dim=-1)
    fs2_predicted = fs2_logits.argmax(dim=-1)
    matches = (hf_predicted == fs2_predicted).float().mean().item()

    print(f"\n  Token match rate: {matches*100:.1f}%")
    print(f"\n  First 5 predictions:")
    print(f"    Pos | HF Token | FS2 Token | Match")
    for i in range(min(5, input_ids.shape[1])):
        hf_tok = hf_predicted[0, i].item()
        fs2_tok = fs2_predicted[0, i].item()
        match = "✓" if hf_tok == fs2_tok else "✗"
        print(f"    {i:3d} | {hf_tok:8d} | {fs2_tok:9d} | {match}")

    print("="*80)
